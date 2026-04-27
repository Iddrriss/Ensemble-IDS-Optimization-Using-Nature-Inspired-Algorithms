import numpy as np
import os
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

BASE_PATH = r"C:\Users\Oxseeker\Desktop\CYS417"

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading UNSW-NB15 data...")
X = np.load(os.path.join(BASE_PATH, "NB15_X.npy"))
y = np.load(os.path.join(BASE_PATH, "NB15_y.npy"))
selected = np.load(os.path.join(BASE_PATH, "NB15_selected_features.npy"))

# ── Split into train/test (80/20) ─────────────────────────────────────────────
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_sel = X_train[:, selected]
X_test_sel  = X_test[:,  selected]

print(f"Train samples : {X_train.shape[0]}")
print(f"Test samples  : {X_test.shape[0]}")
print(f"Features used : {len(selected)} (ensemble selected) out of {X.shape[1]}")

# ════════════════════════════════════════════════════════════════════════════
# MODEL A — Baseline SVM (all 42 features)
# ════════════════════════════════════════════════════════════════════════════
print("\n[Baseline SVM] Training on ALL 42 features...")
start = time.time()
svm_baseline = SVC(kernel="rbf", random_state=42)
svm_baseline.fit(X_train, y_train)
baseline_time = time.time() - start
baseline_acc  = accuracy_score(y_test, svm_baseline.predict(X_test))
print(f"  Accuracy : {baseline_acc*100:.2f}%")
print(f"  Time     : {baseline_time:.2f}s")

# ════════════════════════════════════════════════════════════════════════════
# MODEL B — Optimised SVM (11 ensemble selected features)
# ════════════════════════════════════════════════════════════════════════════
print("\n[Optimised SVM] Training on 11 ensemble-selected features...")
start = time.time()
svm_optimised = SVC(kernel="rbf", random_state=42)
svm_optimised.fit(X_train_sel, y_train)
svm_time = time.time() - start
svm_acc  = accuracy_score(y_test, svm_optimised.predict(X_test_sel))
print(f"  Accuracy : {svm_acc*100:.2f}%")
print(f"  Time     : {svm_time:.2f}s")

# ════════════════════════════════════════════════════════════════════════════
# MODEL C — Random Forest (11 ensemble selected features)
# ════════════════════════════════════════════════════════════════════════════
print("\n[Random Forest] Training on 11 ensemble-selected features...")
start = time.time()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_sel, y_train)
rf_time = time.time() - start
rf_acc  = accuracy_score(y_test, rf_model.predict(X_test_sel))
print(f"  Accuracy : {rf_acc*100:.2f}%")
print(f"  Time     : {rf_time:.2f}s")

# ── Save models and times ─────────────────────────────────────────────────────
joblib.dump(svm_baseline,  os.path.join(BASE_PATH, "NB15_svm_baseline.pkl"))
joblib.dump(svm_optimised, os.path.join(BASE_PATH, "NB15_svm_optimised.pkl"))
joblib.dump(rf_model,      os.path.join(BASE_PATH, "NB15_rf_model.pkl"))

times = {"baseline": baseline_time, "svm": svm_time, "rf": rf_time}
np.save(os.path.join(BASE_PATH, "NB15_training_times.npy"), times)

# ── Save test data for evaluation ─────────────────────────────────────────────
np.save(os.path.join(BASE_PATH, "NB15_X_test.npy"),     X_test)
np.save(os.path.join(BASE_PATH, "NB15_X_test_sel.npy"), X_test_sel)
np.save(os.path.join(BASE_PATH, "NB15_y_test.npy"),     y_test)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Model':<28} {'Accuracy':>10} {'Time':>10}")
print("-"*52)
print(f"{'Baseline SVM (42 feat)':<28} {baseline_acc*100:>9.2f}% {baseline_time:>9.2f}s")
print(f"{'Optimised SVM (11 feat)':<28} {svm_acc*100:>9.2f}% {svm_time:>9.2f}s")
print(f"{'Random Forest (11 feat)':<28} {rf_acc*100:>9.2f}% {rf_time:>9.2f}s")
print("\nAll models saved. Ready for Step 7 evaluation.")