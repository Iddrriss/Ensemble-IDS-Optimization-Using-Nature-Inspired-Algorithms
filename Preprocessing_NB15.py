import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

BASE_PATH  = r"C:\Users\Oxseeker\Desktop\CYS417"
FILE       = os.path.join(BASE_PATH, "UNSW_NB15_training-set.csv")

# ── 1. Load dataset ───────────────────────────────────────────────────────────
print("Loading UNSW-NB15 dataset...")
df = pd.read_csv(FILE)
print(f"Original shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ── 2. Sample exactly 15,000 rows ────────────────────────────────────────────
SAMPLE_SIZE = 15000
df = df.sample(n=SAMPLE_SIZE, random_state=42)
df = df.reset_index(drop=True)
print(f"\nSampled shape: {df.shape}  (random_state=42)")

# ── 3. Convert label to binary (0=normal, 1=attack) ──────────────────────────
# UNSW-NB15 uses 'label' column: 0=normal, 1=attack already
label_col = "label"
print(f"\nLabel distribution:\n{df[label_col].value_counts()}")

# ── 4. Drop non-feature columns ───────────────────────────────────────────────
drop_cols = ["id", "attack_cat", "label"]
drop_cols = [c for c in drop_cols if c in df.columns]
X_df = df.drop(columns=drop_cols)
y    = df[label_col].values

# ── 5. Encode categorical columns ─────────────────────────────────────────────
cat_cols = X_df.select_dtypes(include=["object"]).columns.tolist()
print(f"\nCategorical columns found: {cat_cols}")
le = LabelEncoder()
for col in cat_cols:
    X_df[col] = le.fit_transform(X_df[col].astype(str))

print("Categorical columns encoded.")

# ── 6. Handle any missing values ──────────────────────────────────────────────
X_df.fillna(0, inplace=True)

feature_names = X_df.columns.tolist()
X = X_df.values

# ── 7. Normalize features ─────────────────────────────────────────────────────
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

print(f"\nTotal features   : {X.shape[1]}")
print(f"Total samples    : {X.shape[0]}  ✓ (exactly 15,000)")

# ── 8. Save ───────────────────────────────────────────────────────────────────
np.save(os.path.join(BASE_PATH, "NB15_X.npy"), X)
np.save(os.path.join(BASE_PATH, "NB15_y.npy"), y)

with open(os.path.join(BASE_PATH, "NB15_feature_names.txt"), "w") as f:
    f.write("\n".join(feature_names))

print("\nPreprocessing complete! Files saved:")
print("  NB15_X.npy, NB15_y.npy, NB15_feature_names.txt")
print("\nSampling method: Random sampling with fixed seed (random_state=42)")