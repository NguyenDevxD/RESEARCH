import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from fuzzy_system import FuzzySystem

# --- Chuẩn bị mean + scaler từ dataset ---
file_path = r"D:\computer vision\opencv\myvenv\DATN 2025.1\data_processing\Data\caxton_dataset_filtered.csv"
df = pd.read_csv(file_path)

means = {
    "flow_rate": df[df["flow_rate_class"] == 1]["flow_rate"].mean(),
    "feed_rate": df[df["feed_rate_class"] == 1]["feed_rate"].mean(),
    "z_offset": df[df["z_offset_class"] == 1]["z_offset"].mean(),
    "hotend":   df[df["hotend_class"] == 1]["hotend"].mean()
}

# Fit scaler dựa trên deviation gốc
deviations = pd.DataFrame({
    "flow_rate_deviation": df["flow_rate"] - means["flow_rate"],
    "feed_rate_deviation": df["feed_rate"] - means["feed_rate"],
    "z_offset_deviation":  df["z_offset"] - means["z_offset"],
    "hotend_deviation":    df["hotend"]   - means["hotend"],
})
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(deviations)

def main():
    # Khởi tạo fuzzy system riêng
    fs_flow = FuzzySystem("flow_rate")
    fs_feed = FuzzySystem("feed_rate")
    fs_z    = FuzzySystem("z_offset")
    fs_temp = FuzzySystem("hotend")

    # Giá trị ban đầu
    flow_last = 157
    feed_last = 140
    z_last = -0.05
    temp_last = 220.98

    # Lưu kết quả để vẽ
    results = {
        "flow_rate": [flow_last],
        "feed_rate": [feed_last],
        "z_offset": [z_last],
        "hotend": [temp_last],
    }

    print("step\tflow\tdev_flow\tadj_flow\tfeed\tdev_feed\tadj_feed\tz\tdev_z\tadj_z\ttemp\tdev_temp\tadj_temp")

    n_steps = 50
    for step in range(n_steps):
        # 1. Tính deviation so với mean
        devs = {
            "flow_rate_deviation": flow_last - means["flow_rate"],
            "feed_rate_deviation": feed_last - means["feed_rate"],
            "z_offset_deviation":  z_last - means["z_offset"],
            "hotend_deviation":    temp_last - means["hotend"]
        }

        # 2. Scale deviation về [-1,1] (fix warning)
        dev_df = pd.DataFrame([devs])
        dev_scaled = scaler.transform(dev_df)[0]
        dev_flow, dev_feed, dev_z, dev_temp = dev_scaled

        # 3. Fuzzy inference
        adj_flow = fs_flow.infer(dev_flow)
        adj_feed = fs_feed.infer(dev_feed)
        adj_z    = fs_z.infer(dev_z)
        adj_temp = fs_temp.infer(dev_temp)

        # 4. Update thông số in
        flow_new = flow_last + adj_flow
        feed_new = feed_last + adj_feed
        z_new    = z_last + adj_z
        temp_new = temp_last + adj_temp

        # 5. Lưu kết quả
        results["flow_rate"].append(flow_new)
        results["feed_rate"].append(feed_new)
        results["z_offset"].append(z_new)
        results["hotend"].append(temp_new)

        # 6. In kết quả
        print(f"{step}\t{flow_new:.3f}\t{dev_flow:.3f}\t{adj_flow:.3f}\t"
              f"{feed_new:.3f}\t{dev_feed:.3f}\t{adj_feed:.3f}\t"
              f"{z_new:.3f}\t{dev_z:.3f}\t{adj_z:.3f}\t"
              f"{temp_new:.1f}\t{dev_temp:.3f}\t{adj_temp:.3f}")

        # 7. Cập nhật trạng thái
        flow_last, feed_last, z_last, temp_last = flow_new, feed_new, z_new, temp_new

    # --- Vẽ biểu đồ ---
    steps = range(n_steps + 1)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(steps, results["flow_rate"], marker="o", label="Flow Rate")
    axs[0, 0].axhline(means["flow_rate"], color="r", linestyle="--", label="Mean")
    axs[0, 0].set_title("Flow Rate")
    axs[0, 0].legend()

    axs[0, 1].plot(steps, results["feed_rate"], marker="o", label="Feed Rate")
    axs[0, 1].axhline(means["feed_rate"], color="r", linestyle="--", label="Mean")
    axs[0, 1].set_title("Feed Rate")
    axs[0, 1].legend()

    axs[1, 0].plot(steps, results["z_offset"], marker="o", label="Z Offset")
    axs[1, 0].axhline(means["z_offset"], color="r", linestyle="--", label="Mean")
    axs[1, 0].set_title("Z Offset")
    axs[1, 0].legend()

    axs[1, 1].plot(steps, results["hotend"], marker="o", label="Hotend Temp")
    axs[1, 1].axhline(means["hotend"], color="r", linestyle="--", label="Mean")
    axs[1, 1].set_title("Hotend Temperature")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    print(means)
        # deviation cần scale
    dev_value = np.array([[0.916, 0, 0, 0]])  # đưa vào dạng 2D (1 hàng, 4 cột cho 4 feature)

    scaled_value = scaler.transform(dev_value)
    print("Scaled deviation:", scaled_value[0][0])