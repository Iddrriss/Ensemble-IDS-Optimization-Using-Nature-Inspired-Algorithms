import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             ConfusionMatrixDisplay, roc_curve)
import time

BASE_PATH = r"C:\Users\Oxseeker\Desktop\CYS417"

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data and models...")
X_train  = np.load(os.path.join(BASE_PATH, "X_train.npy"))
X_test   = np.load(os.path.join(BASE_PATH, "X_test.npy"))
y_train  = np.load(os.path.join(BASE_PATH, "y_train.npy"))
y_test   = np.load(os.path.join(BASE_PATH, "y_test.npy"))
selected = np.load(os.path.join(BASE_PATH, "selected_features.npy"))

X_test_sel = X_test[:, selected]

svm_baseline  = joblib.load(os.path.join(BASE_PATH, "svm_baseline.pkl"))
svm_optimised = joblib.load(os.path.join(BASE_PATH, "svm_optimised.pkl"))
rf_model      = joblib.load(os.path.join(BASE_PATH, "rf_model.pkl"))

times = np.load(os.path.join(BASE_PATH, "training_times.npy"), allow_pickle=True).item()

# ── Helper: compute all metrics ───────────────────────────────────────────────
def evaluate(model, X, y, label, exec_time):
    print(f"\n[{label}]")
    start   = time.time()
    y_pred  = model.predict(X)
    inf_time = time.time() - start

    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec  = recall_score(y, y_pred, zero_division=0)
    f1   = f1_score(y, y_pred, zero_division=0)
    cm   = confusion_matrix(y, y_pred)

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # ROC-AUC (decision function or predict_proba)
    try:
        scores = model.decision_function(X)
    except AttributeError:
        scores = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, scores)

    print(f"  Accuracy          : {acc*100:.2f}%")
    print(f"  Precision         : {prec*100:.2f}%")
    print(f"  Recall            : {rec*100:.2f}%")
    print(f"  F1-Score          : {f1*100:.2f}%")
    print(f"  False Positive Rate: {fpr*100:.2f}%")
    print(f"  ROC-AUC           : {auc:.4f}")
    print(f"  Training Time     : {exec_time:.2f}s")
    print(f"  Inference Time    : {inf_time:.2f}s")

    return {
        "label": label, "acc": acc, "prec": prec, "rec": rec,
        "f1": f1, "fpr": fpr, "auc": auc,
        "train_time": exec_time, "inf_time": inf_time,
        "y_pred": y_pred, "scores": scores, "cm": cm
    }

# ── Evaluate all 3 models ─────────────────────────────────────────────────────
print("="*60)
print("FULL EVALUATION")
print("="*60)

r_base = evaluate(svm_baseline,  X_test,     y_test, "Baseline SVM (41 feat)",  times["baseline"])
r_svm  = evaluate(svm_optimised, X_test_sel, y_test, "Optimised SVM (24 feat)", times["svm"])
r_rf   = evaluate(rf_model,      X_test_sel, y_test, "Random Forest (24 feat)", times["rf"])

results = [r_base, r_svm, r_rf]
labels  = [r["label"] for r in results]

# ── Save metrics table ────────────────────────────────────────────────────────
metrics_path = os.path.join(BASE_PATH, "metrics.txt")
with open(metrics_path, "w") as f:
    f.write(f"{'Model':<28} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'FPR':>7} {'AUC':>7} {'Train(s)':>9}\n")
    f.write("-"*85 + "\n")
    for r in results:
        f.write(f"{r['label']:<28} {r['acc']*100:>6.2f}% {r['prec']*100:>6.2f}% "
                f"{r['rec']*100:>6.2f}% {r['f1']*100:>6.2f}% {r['fpr']*100:>6.2f}% "
                f"{r['auc']:>7.4f} {r['train_time']:>9.2f}\n")
print(f"\nMetrics saved to metrics.txt")

# ════════════════════════════════════════════════════════════════════════════
# GRAPHS
# ════════════════════════════════════════════════════════════════════════════
colors = ["#4C72B0", "#DD8452", "#55A868"]

# ── Graph 1: Performance comparison bar chart ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
x = np.arange(len(metric_names))
width = 0.25

for i, r in enumerate(results):
    vals = [r["acc"], r["prec"], r["rec"], r["f1"]]
    bars = ax.bar(x + i*width, [v*100 for v in vals], width, label=r["label"], color=colors[i])
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

ax.set_xlabel("Metric")
ax.set_ylabel("Score (%)")
ax.set_title("Performance Comparison — Baseline vs Optimised Models")
ax.set_xticks(x + width)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 110)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, "graph1_performance.png"), dpi=150)
plt.close()
print("Graph 1 saved: graph1_performance.png")

# ── Graph 2: ROC curves ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for i, r in enumerate(results):
    fpr_c, tpr_c, _ = roc_curve(y_test, r["scores"])
    ax.plot(fpr_c, tpr_c, color=colors[i], lw=2,
            label=f"{r['label']} (AUC={r['auc']:.3f})")
ax.plot([0,1],[0,1], "k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models")
ax.legend(loc="lower right")
ax.grid(linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, "graph2_roc.png"), dpi=150)
plt.close()
print("Graph 2 saved: graph2_roc.png")

# ── Graph 3: Training time comparison ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
train_times = [r["train_time"] for r in results]
bars = ax.bar(labels, train_times, color=colors)
for bar, val in zip(bars, train_times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}s', ha='center', va='bottom', fontsize=10)
ax.set_ylabel("Training Time (seconds)")
ax.set_title("Training Time Comparison")
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, "graph3_time.png"), dpi=150)
plt.close()
print("Graph 3 saved: graph3_time.png")

# ── Graph 4: Confusion matrices ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, r in zip(axes, results):
    disp = ConfusionMatrixDisplay(r["cm"], display_labels=["Normal", "Attack"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(r["label"])
plt.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, "graph4_confusion.png"), dpi=150)
plt.close()
print("Graph 4 saved: graph4_confusion.png")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EVALUATION COMPLETE — FILES SAVED:")
print("="*60)
print("  metrics.txt")
print("  graph1_performance.png")
print("  graph2_roc.png")
print("  graph3_time.png")
print("  graph4_confusion.png")
print("\nReady for Step 7.")