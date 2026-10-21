from controller import fuzzy_controller
from config import clamp


def main():
    current_params = {
        "flow_rate": 100,
        "feed_rate": 60,
        "z_offset": 0.2,
        "hotend": 200
    }

    # Example input from NN
    uniformity = "Bad"
    continuity = "Good"
    adhesion = "Good"

    new_params = fuzzy_controller(uniformity, continuity, adhesion, current_params)
    new_params = clamp(new_params)

    print("Old params:", current_params)
    print("New params:", new_params)


if __name__ == "__main__":
    main()
