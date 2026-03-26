from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
import threading

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "https://mundoevolutivo.netlify.app",
                "http://localhost:5500",
                "http://127.0.0.1:5500"
            ]
        }
    },
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)

MODEL_PATH = "human_tf_model_ptbr.keras"
STATE_SIZE = 19
ACTIONS = [
    "beber água",
    "explorar",
    "caçar",
    "pescar",
    "coletar recursos",
    "construir base",
    "armazenar",
    "descansar",
    "curar",
    "acasalar",
    "ajudar aliado",
    "trocar informações",
    "fugir",
    "reparar base",
    "trocar recursos",
]
ACTION_SIZE = len(ACTIONS)
ACTION_INDEX = {name: idx for idx, name in enumerate(ACTIONS)}
GAMMA = 0.92

# Evita deadlock em chamadas reentrantes com threaded=True
lock = threading.RLock()

BIAS_DECAY = 0.75

cycle_memory = {
    "inherited_cycles": 0,
    "action_bias": {name: 0.0 for name in ACTIONS},
    "lessons": {},
    "mentor_actions": {},
    "last_payload": None,
    "offspring_patterns": {name: 0.0 for name in ACTIONS},
}

_cycles_since_save = 0
SAVE_EVERY_N_CYCLES = 3


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(STATE_SIZE,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(96, activation="relu"),
        tf.keras.layers.Dense(ACTION_SIZE, activation="linear"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
    )
    return model


def load_or_create_model():
    if os.path.exists(MODEL_PATH):
        try:
            loaded = tf.keras.models.load_model(MODEL_PATH)
            input_shape = loaded.input_shape[-1] if isinstance(loaded.input_shape, tuple) else None
            if input_shape == STATE_SIZE:
                return loaded
        except Exception:
            pass
    return create_model()


model = load_or_create_model()

target_model = load_or_create_model()
target_model.set_weights(model.get_weights())
_train_step_counter = 0
TARGET_UPDATE_STEPS = 50


def _state_template(*, health=1.0, satiation=1.0, hydration=1.0, energy=1.0,
                    food=0.3, water=0.3, wood=0.0, stone=0.0, weapon=0.0,
                    base=0.0, age=0.2, wisdom=0.5, courage=0.5,
                    predator=0.0, population=0.5, site=0.0,
                    inherited_knowledge=0.0, parental_bias=0.0, parental_legacy=0.0):
    return np.array([
        health, satiation, hydration, energy,
        food, water, wood, stone,
        weapon, base, age, wisdom,
        courage, predator, population, site,
        inherited_knowledge, parental_bias, parental_legacy,
    ], dtype=np.float32)


PROTOTYPE_STATES = {
    "beber água": _state_template(hydration=0.06, water=0.0, satiation=0.55, energy=0.55),
    "explorar": _state_template(food=0.15, water=0.15, energy=0.75, hydration=0.7, satiation=0.7),
    "caçar": _state_template(food=0.05, water=0.18, weapon=1.0, courage=0.8, hydration=0.7, satiation=0.35),
    "pescar": _state_template(food=0.08, water=0.22, base=0.0, hydration=0.74, satiation=0.4),
    "coletar recursos": _state_template(food=0.1, wood=0.1, stone=0.1, site=0.55, base=0.5, energy=0.68),
    "construir base": _state_template(wood=0.85, stone=0.85, site=0.35, base=0.0, energy=0.72),
    "armazenar": _state_template(food=0.8, water=0.8, wood=0.55, stone=0.55, base=1.0),
    "descansar": _state_template(energy=0.08, base=0.6, hydration=0.52, satiation=0.52, health=0.4),
    "curar": _state_template(health=0.2, energy=0.35, base=1.0),
    "acasalar": _state_template(health=0.9, energy=0.82, hydration=0.82, satiation=0.82, base=1.0, age=0.4),
    "ajudar aliado": _state_template(food=0.7, water=0.7, energy=0.72, population=0.8, base=1.0),
    "trocar informações": _state_template(food=0.45, water=0.45, energy=0.68, population=0.9, wisdom=0.85),
    "fugir": _state_template(predator=1.0, courage=0.35, energy=0.62),
    "reparar base": _state_template(wood=0.4, stone=0.4, base=1.0, site=0.4, energy=0.65),
    "trocar recursos": _state_template(food=0.5, water=0.5, wood=0.5, stone=0.5, base=1.0, population=0.85),
}


def _apply_state_conditioned_bias(states: np.ndarray, q_values: np.ndarray):
    """
    Bias contextual aplicado APENAS na inferencia (decide_batch).
    FIX 2: NAO chamar dentro do train_batch — senao o target Bellman fica inflado.
    """
    action_bias = cycle_memory.get("action_bias", {})
    lessons = cycle_memory.get("lessons", {})
    mentor_actions = cycle_memory.get("mentor_actions", {})
    for i, state in enumerate(states):
        health, satiation, hydration, energy = state[0], state[1], state[2], state[3]
        food, water, wood, stone = state[4], state[5], state[6], state[7]
        has_weapon, has_base, age = state[8], state[9], state[10]
        wisdom, courage, predator, population, site = state[11], state[12], state[13], state[14], state[15]
        inherited_knowledge, parental_bias, parental_legacy = state[16], state[17], state[18]

        for action_name, bias in action_bias.items():
            idx = ACTION_INDEX[action_name]
            q_values[i, idx] += float(bias)

        if hydration < 0.25:
            q_values[i, ACTION_INDEX["beber água"]] += 2.6 + float(lessons.get("waterUrgency", 0)) * 0.18
        if satiation < 0.3:
            q_values[i, ACTION_INDEX["coletar recursos"]] += 1.5 + float(lessons.get("foodUrgency", 0)) * 0.16
            q_values[i, ACTION_INDEX["caçar"]] += 0.6 + has_weapon * 0.8
        if energy < 0.18:
            q_values[i, ACTION_INDEX["descansar"]] += 2.4 + has_base * 0.6
        if health < 0.35:
            q_values[i, ACTION_INDEX["curar"]] += 2.4
        if predator > 0.5:
            q_values[i, ACTION_INDEX["fugir"]] += 3.0 + (1.0 - courage) * 0.8
        if site > 0.2 and wood < 0.35:
            q_values[i, ACTION_INDEX["coletar recursos"]] += 1.2 + float(lessons.get("baseUrgency", 0)) * 0.12
        if site > 0.2 and stone < 0.35:
            q_values[i, ACTION_INDEX["coletar recursos"]] += 1.2 + float(lessons.get("baseUrgency", 0)) * 0.12
        if has_base > 0.5 and (wood > 0.35 or stone > 0.35 or food > 0.5 or water > 0.5):
            q_values[i, ACTION_INDEX["armazenar"]] += 0.8
        if wood > 0.7 and stone > 0.7 and has_base < 0.5:
            q_values[i, ACTION_INDEX["construir base"]] += 1.9 + float(lessons.get("baseUrgency", 0)) * 0.14
        if population > 0.7 and wisdom > 0.55:
            q_values[i, ACTION_INDEX["trocar informações"]] += 0.18 + float(lessons.get("shareUrgency", 0)) * 0.04
            q_values[i, ACTION_INDEX["ajudar aliado"]] += 0.22
        if has_weapon > 0.5 and courage > 0.55 and satiation < 0.6:
            q_values[i, ACTION_INDEX["caçar"]] += 1.0
        if hydration > 0.45 and satiation > 0.45 and energy > 0.5 and age > 0.2:
            q_values[i, ACTION_INDEX["acasalar"]] += float(lessons.get("reproductionUrgency", 0)) * 0.15

        if inherited_knowledge > 0.18:
            q_values[i, ACTION_INDEX["trocar informações"]] += inherited_knowledge * 1.1
            q_values[i, ACTION_INDEX["explorar"]] += inherited_knowledge * 0.45
        if parental_bias > 0.2:
            q_values[i, ACTION_INDEX["explorar"]] += parental_bias * 0.35
            q_values[i, ACTION_INDEX["armazenar"]] += parental_bias * 0.22 * has_base
        if parental_legacy > 0.15:
            q_values[i, ACTION_INDEX["construir base"]] += parental_legacy * 0.4 * (1.0 - has_base)
            q_values[i, ACTION_INDEX["ajudar aliado"]] += parental_legacy * 0.25

        offspring_patterns = cycle_memory.get("offspring_patterns", {})
        for action_name, boost in offspring_patterns.items():
            idx = ACTION_INDEX.get(action_name)
            if idx is not None:
                q_values[i, idx] += float(boost) * (0.08 + inherited_knowledge * 0.12 + parental_bias * 0.08)

        for action_name, boost in mentor_actions.items():
            idx = ACTION_INDEX.get(action_name)
            if idx is not None:
                q_values[i, idx] += float(boost) * (0.12 + wisdom * 0.08)


def _train_on_cycle_knowledge(payload: dict):
    global _cycles_since_save

    archive = payload.get("archive", {}) or {}
    run_summary = payload.get("run_summary", {}) or {}
    lessons = archive.get("lessons", {}) or {}
    action_totals = archive.get("action_totals", {}) or run_summary.get("action_totals", {}) or {}
    mentors = archive.get("mentors", []) or []

    old_bias = cycle_memory.get("action_bias", {})
    cycle_memory["inherited_cycles"] = int(archive.get("cycles", 0) or payload.get("ended_cycle", 0) or 0)
    cycle_memory["lessons"] = {k: float(v or 0) for k, v in lessons.items()}
    cycle_memory["last_payload"] = payload
    cycle_memory["action_bias"] = {name: old_bias.get(name, 0.0) * BIAS_DECAY for name in ACTIONS}
    cycle_memory["mentor_actions"] = {name: 0.0 for name in ACTIONS}

    max_total = max([float(v or 0) for v in action_totals.values()] + [1.0])
    for action_name, total in action_totals.items():
        idx = ACTION_INDEX.get(action_name)
        if idx is None:
            continue
        new_val = min(1.8, (float(total) / max_total) * 1.6)
        cycle_memory["action_bias"][action_name] = min(2.0, cycle_memory["action_bias"][action_name] + new_val)

    for mentor in mentors:
        legacy = float(mentor.get("legacyScore", 0) or 0)
        score = float(mentor.get("score", 0) or 0)
        weight = min(2.0, 0.15 + (legacy * 0.03) + (score * 0.015))
        for action in mentor.get("dominantActions", []) or []:
            action_name = action.get("action") or action.get("label")
            if action_name in ACTION_INDEX:
                cycle_memory["mentor_actions"][action_name] += weight
                cycle_memory["action_bias"][action_name] = min(2.4, cycle_memory["action_bias"][action_name] + weight * 0.35)

    states = []
    targets = []
    current_q = model.predict(np.stack(list(PROTOTYPE_STATES.values())), verbose=0)
    prototype_names = list(PROTOTYPE_STATES.keys())
    name_to_row = {name: row for name, row in zip(prototype_names, current_q)}

    def push_sample(action_name: str, strength: float):
        proto = PROTOTYPE_STATES.get(action_name)
        if proto is None:
            return
        base_target = np.array(name_to_row[action_name], copy=True)
        idx = ACTION_INDEX[action_name]
        base_target[idx] = max(base_target[idx], float(strength))
        states.append(proto)
        targets.append(base_target)

    for action_name, total in action_totals.items():
        if action_name in ACTION_INDEX:
            normalized = float(total or 0) / max_total
            push_sample(action_name, 1.0 + normalized * 3.0)

    lesson_to_action = {
        "waterUrgency": "beber água",
        "foodUrgency": "coletar recursos",
        "baseUrgency": "construir base",
        "weaponUrgency": "caçar",
        "shareUrgency": "trocar informações",
        "reproductionUrgency": "acasalar",
        "longevityUrgency": "descansar",
    }
    for lesson_key, action_name in lesson_to_action.items():
        strength = float(lessons.get(lesson_key, 0) or 0)
        if strength > 0:
            push_sample(action_name, 1.2 + min(3.2, strength * 0.22))

    for mentor in mentors:
        base = 0.8 + min(2.8, float(mentor.get("legacyScore", 0) or 0) * 0.05)
        for action in mentor.get("dominantActions", []) or []:
            action_name = action.get("action") or action.get("label")
            if action_name in ACTION_INDEX:
                push_sample(action_name, base)

    if not states:
        return 0, 0.0

    states_np = np.stack(states).astype(np.float32)
    targets_np = np.stack(targets).astype(np.float32)
    history = model.fit(states_np, targets_np, epochs=4, verbose=0)
    loss = float(history.history.get("loss", [0.0])[-1])

    _cycles_since_save += 1
    if _cycles_since_save >= SAVE_EVERY_N_CYCLES:
        model.save(MODEL_PATH)
        _cycles_since_save = 0

    return len(states_np), loss


def _train_on_offspring_knowledge(payload: dict):
    child = payload.get("child", {}) or {}
    inherited_knowledge = float(child.get("inherited_knowledge", 0) or 0)
    parental_legacy_score = float(child.get("parental_legacy_score", 0) or 0)
    parental_action_bias = child.get("parental_action_bias", {}) or {}
    dominant_actions = child.get("dominant_actions", []) or []

    norm_knowledge = float(np.clip(inherited_knowledge / 100.0, 0.0, 1.0))
    norm_legacy = float(np.clip(parental_legacy_score / 50.0, 0.0, 1.0))
    strongest_bias = max([float(v or 0) for v in parental_action_bias.values()] + [0.0])
    norm_bias = float(np.clip(strongest_bias, 0.0, 1.0))

    cycle_memory["offspring_patterns"] = {name: 0.0 for name in ACTIONS}
    for action_name, bias in parental_action_bias.items():
        if action_name in ACTION_INDEX:
            cycle_memory["offspring_patterns"][action_name] = max(
                cycle_memory["offspring_patterns"][action_name],
                float(np.clip(bias, 0.0, 1.5))
            )

    current_q = model.predict(np.stack(list(PROTOTYPE_STATES.values())), verbose=0)
    prototype_names = list(PROTOTYPE_STATES.keys())
    name_to_row = {name: row for name, row in zip(prototype_names, current_q)}

    ranked_actions = []
    for action_name, bias in parental_action_bias.items():
        if action_name in ACTION_INDEX:
            ranked_actions.append((action_name, float(bias or 0)))
    for item in dominant_actions:
        action_name = item.get("action") if isinstance(item, dict) else None
        strength = float(item.get("strength", 0) or 0) if isinstance(item, dict) else 0.0
        if action_name in ACTION_INDEX:
            ranked_actions.append((action_name, max(0.2, strength)))
    ranked_actions.sort(key=lambda pair: pair[1], reverse=True)
    ranked_actions = ranked_actions[:4]

    if not ranked_actions:
        return 0, 0.0

    states = []
    targets = []
    for action_name, strength in ranked_actions:
        proto = np.array(PROTOTYPE_STATES[action_name], copy=True)
        proto[16] = norm_knowledge
        proto[17] = norm_bias
        proto[18] = norm_legacy
        target = np.array(name_to_row[action_name], copy=True)
        target[ACTION_INDEX[action_name]] = max(
            target[ACTION_INDEX[action_name]],
            1.2 + float(strength) * 2.8 + norm_knowledge * 0.6
        )
        states.append(proto)
        targets.append(target)

    states_np = np.stack(states).astype(np.float32)
    targets_np = np.stack(targets).astype(np.float32)
    history = model.fit(states_np, targets_np, epochs=2, verbose=0)
    return len(states_np), float(history.history.get("loss", [0.0])[-1])


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status": "running",
        "model_loaded": True,
        "state_size": STATE_SIZE,
        "action_size": ACTION_SIZE,
        "actions": ACTIONS,
        "language": "pt-BR",
        "model_path": MODEL_PATH,
        "inherited_cycles": cycle_memory.get("inherited_cycles", 0),
        "train_steps": _train_step_counter,
        "target_network_syncs": _train_step_counter // TARGET_UPDATE_STEPS,
    })


@app.route("/config", methods=["GET"])
def config():
    return jsonify({
        "state_size": STATE_SIZE,
        "actions": ACTIONS,
        "gamma": GAMMA,
        "language": "pt-BR",
    })


@app.route("/decide_batch", methods=["POST", "OPTIONS"])
def decide_batch():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(force=True) or {}
    states = np.array(data.get("states", []), dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != STATE_SIZE:
        return jsonify({"error": f"states must be [N, {STATE_SIZE}]"}), 400

    with lock:
        q_values = model.predict(states, verbose=0)

    _apply_state_conditioned_bias(states, q_values)
    actions = np.argmax(q_values, axis=1).astype(int).tolist()
    action_names = [ACTIONS[idx] for idx in actions]
    return jsonify({
        "actions": actions,
        "action_names": action_names,
        "q_values": q_values.tolist(),
    })


@app.route("/train_batch", methods=["POST", "OPTIONS"])
def train_batch():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(force=True) or {}
    states = np.array(data.get("states", []), dtype=np.float32)
    actions = np.array(data.get("actions", []), dtype=np.int32)
    rewards = np.array(data.get("rewards", []), dtype=np.float32)
    next_states = np.array(data.get("next_states", []), dtype=np.float32)
    dones = np.array(data.get("dones", []), dtype=np.float32)

    if states.ndim != 2 or states.shape[1] != STATE_SIZE:
        return jsonify({"error": f"states must be [N, {STATE_SIZE}]"}), 400
    if next_states.ndim != 2 or next_states.shape[1] != STATE_SIZE:
        return jsonify({"error": f"next_states must be [N, {STATE_SIZE}]"}), 400
    batch_size = len(states)
    if not (len(actions) == len(rewards) == len(next_states) == len(dones) == batch_size):
        return jsonify({"error": "states/actions/rewards/next_states/dones sizes must match"}), 400

    with lock:
        current_q = model.predict(states, verbose=0)
        next_q = target_model.predict(next_states, verbose=0)
        targets = np.array(current_q, copy=True)
        for i in range(batch_size):
            action_idx = int(actions[i])
            if 0 <= action_idx < ACTION_SIZE:
                future = 0.0 if dones[i] >= 1 else float(np.max(next_q[i]))
                targets[i, action_idx] = float(rewards[i]) + GAMMA * future
        history = model.fit(states, targets, epochs=1, verbose=0)
        loss = float(history.history.get("loss", [0.0])[-1])

        global _train_step_counter
        _train_step_counter += 1
        if _train_step_counter % TARGET_UPDATE_STEPS == 0:
            target_model.set_weights(model.get_weights())
    return jsonify({"status": "trained", "batch_size": batch_size, "loss": loss})


@app.route("/inherit_cycle", methods=["POST", "OPTIONS"])
def inherit_cycle():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(force=True) or {}
    with lock:
        sample_count, loss = _train_on_cycle_knowledge(payload)
    return jsonify({
        "status": "ok",
        "message": "Memória ancestral incorporada ao TensorFlow.",
        "synthetic_samples": sample_count,
        "loss": loss,
        "inherited_cycles": cycle_memory.get("inherited_cycles", 0),
    })


@app.route("/inherit_offspring", methods=["POST", "OPTIONS"])
def inherit_offspring():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(force=True) or {}
    with lock:
        sample_count, loss = _train_on_offspring_knowledge(payload)
    return jsonify({
        "status": "ok",
        "message": "Herança micro incorporada ao TensorFlow.",
        "synthetic_samples": sample_count,
        "loss": loss,
    })


@app.route("/save", methods=["POST", "OPTIONS"])
def save_model():
    if request.method == "OPTIONS":
        return ("", 204)

    with lock:
        model.save(MODEL_PATH)
    return jsonify({"status": "saved", "path": MODEL_PATH})


@app.route("/reset", methods=["POST", "OPTIONS"])
def reset_model():
    if request.method == "OPTIONS":
        return ("", 204)

    global model, _cycles_since_save, target_model, _train_step_counter
    with lock:
        model = create_model()
        target_model = create_model()
        target_model.set_weights(model.get_weights())
        _cycles_since_save = 0
        _train_step_counter = 0
        cycle_memory["inherited_cycles"] = 0
        cycle_memory["action_bias"] = {name: 0.0 for name in ACTIONS}
        cycle_memory["lessons"] = {}
        cycle_memory["mentor_actions"] = {name: 0.0 for name in ACTIONS}
        cycle_memory["last_payload"] = None
        cycle_memory["offspring_patterns"] = {name: 0.0 for name in ACTIONS}
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Servidor TensorFlow dos humanos em http://0.0.0.0:{port}")
    # Para producao: gunicorn -w 1 --timeout 120 server:app
    app.run(host="0.0.0.0", port=port, threaded=True)
