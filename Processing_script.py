import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

# ── 1. Column names for NSL-KDD ──────────────────────────────────────────────
col_names = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

# ── 2. File paths ────────────────────────────────────────────────────────────
BASE_PATH  = r"C:\Users\Oxseeker\Desktop\CYS417"
TRAIN_FILE = os.path.join(BASE_PATH, "KDDTrain+.txt")
TEST_FILE  = os.path.join(BASE_PATH, "KDDTest+.txt")

# ── 3. Load datasets ─────────────────────────────────────────────────────────
print("Loading datasets...")
train_df = pd.read_csv(TRAIN_FILE, header=None, names=col_names)
test_df  = pd.read_csv(TEST_FILE,  header=None, names=col_names)
print(f"Original train shape : {train_df.shape}")
print(f"Original test shape  : {test_df.shape}")

# ── 4. Sample exactly 20,000 rows from training set ──────────────────────────
SAMPLE_SIZE = 20000
train_df = train_df.sample(n=SAMPLE_SIZE, random_state=42)
train_df = train_df.reset_index(drop=True)
print(f"\nSampled train shape  : {train_df.shape}  (random_state=42)")

# ── 5. Drop difficulty column ────────────────────────────────────────────────
train_df.drop(columns=["difficulty"], inplace=True)
test_df.drop(columns=["difficulty"],  inplace=True)

# ── 6. Convert label to binary (normal=0, attack=1) ──────────────────────────
train_df["label"] = train_df["label"].apply(lambda x: 0 if x == "normal" else 1)
test_df["label"]  = test_df["label"].apply(lambda x: 0 if x == "normal" else 1)
print(f"\nTrain label distribution:\n{train_df['label'].value_counts()}")
print(f"\nTest label distribution:\n{test_df['label'].value_counts()}")

# ── 7. Encode categorical columns ────────────────────────────────────────────
cat_cols = ["protocol_type", "service", "flag"]
le = LabelEncoder()

for col in cat_cols:
    combined = pd.concat([train_df[col], test_df[col]])
    le.fit(combined)
    train_df[col] = le.transform(train_df[col])
    test_df[col]  = le.transform(test_df[col])

print("\nCategorical columns encoded.")

# ── 8. Separate features and labels ──────────────────────────────────────────
X_train = train_df.drop(columns=["label"]).values
y_train = train_df["label"].values

X_test  = test_df.drop(columns=["label"]).values
y_test  = test_df["label"].values

feature_names = train_df.drop(columns=["label"]).columns.tolist()

# ── 9. Normalize features (scale to 0-1) ─────────────────────────────────────
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print("\nFeatures normalized.")
print(f"\nTotal features     : {X_train.shape[1]}")
print(f"Training samples   : {X_train.shape[0]}  ✓ (exactly 20,000)")
print(f"Testing samples    : {X_test.shape[0]}")

# ── 10. Save preprocessed data ───────────────────────────────────────────────
np.save(os.path.join(BASE_PATH, "X_train.npy"), X_train)
np.save(os.path.join(BASE_PATH, "X_test.npy"),  X_test)
np.save(os.path.join(BASE_PATH, "y_train.npy"), y_train)
np.save(os.path.join(BASE_PATH, "y_test.npy"),  y_test)

with open(os.path.join(BASE_PATH, "feature_names.txt"), "w") as f:
    f.write("\n".join(feature_names))

print("\nPreprocessing complete! Files saved:")
print("  X_train.npy, X_test.npy, y_train.npy, y_test.npy, feature_names.txt")
print("\nSampling method: Random sampling with fixed seed (random_state=42)")