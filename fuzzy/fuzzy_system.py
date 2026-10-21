import numpy as np
from typing import Dict
from fuzzy.membership_function import trimf


ADJUSTMENT_DEADZONE = 0.05

# Defuzzification resolution
DEFUZZ_X_SAMPLES = 201


# ============================
# Membership functions (per parameter)
# ============================

DEVIATION_MF = {
    "flow_rate": {
        "Very Low":   (20, 20, 55),
        "Low":        (20, 55, 90),
        "Medium":     (55, 90, 131),
        "Large":      (90, 131, 166),
        "Very Large": (131, 200, 200)
    },
    "feed_rate": {
        "Very Low":   (20, 20, 55),
        "Low":        (20, 55, 90),
        "Medium":     (55, 90, 131),
        "Large":      (90, 131, 166),
        "Very Large": (131, 200, 200)
    },
    "z_offset": {
        "Very Low":   (-0.08, -0.08, -0.05),
        "Low":        (-0.08, -0.05, -0.02),
        "Medium":     (-0.05, -0.02, 0.06),
        "Large":      (-0.02, 0.06, 0.09),
        "Very Large": (0.06, 0.12, 0.12 )
    },
    "hotend": {
        "Very Low":   (167.47, 167.47, 183.75),
        "Low":        (167.47, 183.75, 200.00),
        "Medium":     (183.75, 200.00, 220.04),
        "Large":      (200.00, 220.04, 230.06),
        "Very Large": (220.04, 251.80, 251.80)
    }
}

ADJUST_MF = {
    "flow_rate": {
        "Positive High": (5, 10, 10),
        "Positive":      (0, 5, 10),
        "Normal":        (-5, 0, 5),
        "Negative":      (-10, -5, 0),
        "Negative High": (-10, -10, -5)
    },
    "feed_rate": {
        "Positive High": (5, 10, 10),
        "Positive":      (0, 5, 10),
        "Normal":        (-5, 0, 5),
        "Negative":      (-10, -5, 0),
        "Negative High": (-10, -10, -5)
    },
    "z_offset": {
        "Positive High": (0.05, 0.1, 0.1),
        "Positive":      (0.0, 0.05, 0.1),
        "Normal":        (-0.05, 0.0, 0.05),
        "Negative":      (-0.1, -0.05, 0.0),
        "Negative High": (-0.1, -0.1, -0.05)
    },
    "hotend": {
        "Positive High": (5, 10, 10),
        "Positive":      (0, 5, 10),
        "Normal":        (-5, 0, 5),
        "Negative":      (-10, -5, 0),
        "Negative High": (-10, -10, -5)
    }
}

# ============================
# Rules (per parameter)
# ============================

RULES = {
    "flow_rate": [
        ("Very Low", "Positive High"),
        ("Low", "Positive"),
        ("Medium", "Normal"),
        ("Large", "Negative"),
        ("Very Large", "Negative High")
    ],
    "feed_rate": [
        ("Very Low", "Positive High"),
        ("Low", "Positive"),
        ("Medium", "Normal"),
        ("Large", "Negative"),
        ("Very Large", "Negative High")
    ],
    "z_offset": [
        ("Very Low", "Positive High"),
        ("Low", "Positive"),
        ("Medium", "Normal"),
        ("Large", "Negative"),
        ("Very Large", "Negative High")
    ],
    "hotend": [
        ("Very Low", "Positive High"),
        ("Low", "Positive"),
        ("Medium", "Normal"),
        ("Large", "Negative"),
        ("Very Large", "Negative High")
    ]
}

# ============================
# Fuzzy System
# ============================

class FuzzySystem:
    def __init__(self, param_type: str):
        if param_type not in DEVIATION_MF:
            raise ValueError(f"Unknown parameter type: {param_type}")

        self.param_type = param_type
        self.dev_mf = DEVIATION_MF[param_type]
        self.adj_mf_params = ADJUST_MF[param_type]
        self.rules = RULES[param_type]

        # Universe for defuzzification
        min_x = min(v[0] for v in self.adj_mf_params.values())
        max_x = max(v[2] for v in self.adj_mf_params.values())
        self.x = np.linspace(min_x, max_x, DEFUZZ_X_SAMPLES)

        # Precompute adjustment MFs
        self.adj_mfs = {k: trimf(self.x, v) for k, v in self.adj_mf_params.items()}

    def fuzzify_deviation(self, val: float) -> Dict[str, float]:
        return {k: float(trimf(np.array([val]), params)[0]) for k, params in self.dev_mf.items()}

    def infer(self, deviation: float) -> float:
        """ Mamdani inference + centroid defuzzification """
        fuzz_dev = self.fuzzify_deviation(deviation)

        aggregated = np.zeros_like(self.x)
        for dev_label, out_label in self.rules:
            firing = fuzz_dev.get(dev_label, 0.0)
            if firing > 0:
                clipped = np.minimum(firing, self.adj_mfs[out_label])
                aggregated = np.maximum(aggregated, clipped)

        # centroid - defuzzification
        area = np.sum(aggregated)
        crisp = np.sum(self.x * aggregated) / area if area > 0 else 0.0

        if abs(crisp) < ADJUSTMENT_DEADZONE:
            return 0.0

        return float(crisp)
