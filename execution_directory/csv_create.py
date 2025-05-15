import pandas as pd

# Load feature CSVs
back_df = pd.read_csv("back_features.csv")
side_df = pd.read_csv("side_features.csv")

# Ensure they each have exactly one row
if len(back_df) != 1 or len(side_df) != 1:
    raise ValueError("Each CSV should contain exactly one row of features.")

# Merge them horizontally (side-by-side)
merged_df = pd.concat([back_df.reset_index(drop=True), side_df.reset_index(drop=True)], axis=1)

# Add extra body measurements
merged_df["Height (in cm)"] = 175   # <-- Change as needed
merged_df["Inseam Length (in cm)"] = 80   # <-- Change as needed
merged_df["Arm Length (in cm)"] = 65   # <-- Change as needed

# Save to new CSV
merged_df.to_csv("13_samples_input.csv", index=False)
print("Merged data saved to 13_samples_input.csv")
