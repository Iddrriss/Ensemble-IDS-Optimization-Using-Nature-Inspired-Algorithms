import numpy as np
import os
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

BASE_PATH = r"C:\Users\Oxseeker\Desktop\CYS417"

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
X_train = np.load(os.path.join(BASE_PATH, "X_train.npy"))
X_test  = np.load(os.path.join(BASE_PATH, "X_test.npy"))
y_train = np.load(os.path.join(BASE_PATH, "y_train.npy"))
y_test  = np.load(os.path.join(BASE_PATH, "y_test.npy"))

selected = np.load(os.path.join(BASE_PATH, "selected_features.npy"))

# ── Slice to selected features only ──────────────────────────────────────────
X_train_sel = X_train[:, selected]
X_test_sel  = X_test[:,  selected]

print(f"Training with {len(selected)} selected features out of {X_train.shape[1]}")

# ════════════════════════════════════════════════════════════════════════════
# MODEL A — Baseline SVM  (no feature selection, all 41 features)
# ════════════════════════════════════════════════════════════════════════════
print("\n[Baseline SVM] Training on ALL 41 features...")
start = time.time()
svm_baseline = SVC(kernel="rbf", random_state=42)
svm_baseline.fit(X_train, y_train)
baseline_time = time.time() - start

baseline_acc = accuracy_score(y_test, svm_baseline.predict(X_test))
print(f"  Accuracy : {baseline_acc*100:.2f}%")
print(f"  Time     : {baseline_time:.2f}s")

# ════════════════════════════════════════════════════════════════════════════
# MODEL B — Optimised SVM  (ensemble selected 14 features)
# ════════════════════════════════════════════════════════════════════════════
print("\n[Optimised SVM] Training on 14 ensemble-selected features...")
start = time.time()
svm_optimised = SVC(kernel="rbf", random_state=42)
svm_optimised.fit(X_train_sel, y_train)
svm_time = time.time() - start

svm_acc = accuracy_score(y_test, svm_optimised.predict(X_test_sel))
print(f"  Accuracy : {svm_acc*100:.2f}%")
print(f"  Time     : {svm_time:.2f}s")

# ════════════════════════════════════════════════════════════════════════════
# MODEL C — Random Forest  (ensemble selected 14 features)
# ════════════════════════════════════════════════════════════════════════════
print("\n[Random Forest] Training on 14 ensemble-selected features...")
start = time.time()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_sel, y_train)
rf_time = time.time() - start

rf_acc = accuracy_score(y_test, rf_model.predict(X_test_sel))
print(f"  Accuracy : {rf_acc*100:.2f}%")
print(f"  Time     : {rf_time:.2f}s")

# ── Save all 3 models ─────────────────────────────────────────────────────────
joblib.dump(svm_baseline,  os.path.join(BASE_PATH, "svm_baseline.pkl"))
joblib.dump(svm_optimised, os.path.join(BASE_PATH, "svm_optimised.pkl"))
joblib.dump(rf_model,      os.path.join(BASE_PATH, "rf_model.pkl"))

# ── Save timing for later use in report ──────────────────────────────────────
times = {"baseline": baseline_time, "svm": svm_time, "rf": rf_time}
np.save(os.path.join(BASE_PATH, "training_times.npy"), times)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Model':<25} {'Accuracy':>10} {'Time':>10}")
print("-"*50)
print(f"{'Baseline SVM (41 feat)':<25} {baseline_acc*100:>9.2f}% {baseline_time:>9.2f}s")
print(f"{'Optimised SVM (14 feat)':<25} {svm_acc*100:>9.2f}% {svm_time:>9.2f}s")
print(f"{'Random Forest (14 feat)':<25} {rf_acc*100:>9.2f}% {rf_time:>9.2f}s")

print("\nAll models saved. Ready for Step 6.")