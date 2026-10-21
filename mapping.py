import pandas as pd

# flow_rate_class, feed_rate_class, z_offset_class, hotend_class
# 0 = low, 1 = medium, 2 = high

rules = [

    # uniformity=1 continuity=1 adhesion=1
    (1,1,1,1, 1,1,1),
    (1,2,1,1, 1,1,1),
    (1,2,1,2, 1,1,1),
    (1,1,1,2, 1,1,1),

    # uniformity=1 continuity=1 adhesion=0
    (1,1,2,1, 1,1,0),
    (1,1,2,2, 1,1,0),
    (1,1,2,0, 1,1,0),
    (1,0,2,1, 1,1,0),
    (1,2,2,1, 1,1,0),

    # uniformity = 1 continuity=0 adhesion=1
    (0,2,1,0, 1,0,1),
    (0,2,1,1, 1,0,1),
    (1,2,1,0, 1,0,1),

    # uniformity=1 continuity=0 adhesion=0
    (1,2,2,0, 1,0,0),
    (0,2,2,0, 1,0,0),
    (0,2,2,1, 1,0,0),

    # uniformity=0 continuity=1 adhesion=1
    (2,0,0,1, 0,1,1),
    (2,0,0,2, 0,1,1),
    (2,0,1,2, 0,1,1),

    # uniformity=0 continuity=1 adhesion=0
    (2,0,2,1, 0,1,0),
    (2,0,2,2, 0,1,0),
    (1,0,2,2, 0,1,0),

    # uniformity=0 continuity=0 adhesion=1
    (0,2,0,1, 0,0,1),
    (0,2,0,0, 0,0,1),
    (1,2,0,0, 0,0,1),

]

columns = [
    "flow_rate_class",
    "feed_rate_class",
    "z_offset_class",
    "hotend_class",
    "uniformity",
    "continuity",
    "adhesion"
]

df = pd.DataFrame(rules, columns=columns)

df.to_csv("rule_mapping_24.csv", index=False)

print("Saved rule_mapping_24.csv")

'''
import pandas as pd

# đọc file csv mapping
df = pd.read_csv("fdm_mapping_final.csv")

# tạo key cho 3 đặc tính
df["state"] = (
    df["uniformity"].astype(str) +
    df["continuity"].astype(str) +
    df["adhesion"].astype(str)
)

# nhóm theo trạng thái
groups = df.groupby("state")

for state, group in groups:

    print("\n====================================")
    print(f"STATE {state}  (U,C,A)")
    print("Number of combinations:", len(group))
    print("------------------------------------")

    for _, row in group.iterrows():
        print(
            f"flow={row['flow_rate_class']}, "
            f"feed={row['feed_rate_class']}, "
            f"z={row['z_offset_class']}, "
            f"temp={row['hotend_class']}"
        )
        '''