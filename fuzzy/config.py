LIMITS = {       #MIN-MAX params
    "flow_rate": (20, 200),
    "feed_rate": (20, 200),
    "z_offset": (-1.0, 3.0),
    "hotend": (180, 260)
}


def clamp(params):
    new_params = params.copy()
    for key in params:
        low, high = LIMITS[key]
        new_params[key] = max(low, min(high, params[key]))
    return new_params
