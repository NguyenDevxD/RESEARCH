# =========================
# file: fuzzy_rules.py
# =========================

def get_actions(uniformity, continuity, adhesion):
    if uniformity == "Good" and continuity == "Good" and adhesion == "Good":
        return ["Remain", "Remain", "Remain", "Remain"]

    elif uniformity == "Bad" and continuity == "Good" and adhesion == "Good":
        return ["Decrease", "Increase", "Remain", "Remain"]

    elif uniformity == "Good" and continuity == "Bad" and adhesion == "Good":
        return ["Increase", "Decrease", "Remain", "Remain"]

    elif uniformity == "Good" and continuity == "Good" and adhesion == "Bad":
        return ["Remain", "Remain", "Decrease", "Increase"]

    elif uniformity == "Bad" and continuity == "Bad" and adhesion == "Good":
        return ["Increase", "Remain", "Increase", "Remain"]

    elif uniformity == "Bad" and continuity == "Good" and adhesion == "Bad":
        return ["Decrease", "Remain", "Decrease", "Increase"]

    elif uniformity == "Good" and continuity == "Bad" and adhesion == "Bad":
        return ["Increase", "Decrease", "Decrease", "Increase"]

    else:
        raise ValueError("Invalid input combination")
