from fuzzy.fuzzy_rules import get_actions


def adjust(value, action, step):
    if action == "Increase":
        return value + step
    elif action == "Decrease":
        return value - step
    return value


def fuzzy_controller(uniformity, continuity, adhesion, params):
    actions = get_actions(uniformity, continuity, adhesion)

    new_params = params.copy()

    new_params["flow_rate"] = adjust(params["flow_rate"], actions[0], 5)
    new_params["feed_rate"] = adjust(params["feed_rate"], actions[1], 5)
    new_params["z_offset"] = adjust(params["z_offset"], actions[2], 0.01)
    new_params["hotend"] = adjust(params["hotend"], actions[3], 5)

    return new_params