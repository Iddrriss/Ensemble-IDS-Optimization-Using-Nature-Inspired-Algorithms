# CYS 417 – Ensemble IDS Optimization
**Shofadekan Idris Oladapo | CYS/22/9093**

---

## What This Project Does

This project builds a system that can detect network attacks automatically.
It looks at network traffic data and decides: is this **normal** or is this an **attack**?

To do that, it uses three nature-inspired algorithms working as a team to pick
the most important clues (features) from the data, then trains a classifier to
make the final decision.

---

## Files in This Folder

| File | What it does |
|---|---|
| `Processing_script.py` | Loads and cleans the dataset |
| `ensemble_features.py` | Runs PSO, GA, and GWO to select the best features |
| `classifiers.py` | Trains the SVM and Random Forest models |
| `step6_evaluate.py` | Tests the models and generates all graphs |
| `step7_preprocess_NB15.py` | Cleans the UNSW-NB15 dataset |
| `step7_ensemble_NB15.py` | Runs feature selection on UNSW-NB15 |
| `step7_classifiers_NB15.py` | Trains models on UNSW-NB15 |
| `step7_evaluate_NB15.py` | Tests models on UNSW-NB15 and generates graphs |
| `KDDTrain+.txt` | NSL-KDD training data |
| `KDDTest+.txt` | NSL-KDD test data |
| `UNSW_NB15_training-set.csv` | UNSW-NB15 dataset |

---

## What You Need Before Running

- Python 3.11 or higher
- The following libraries (install once with the command below):

```
pip install numpy pandas scikit-learn matplotlib joblib
```

---

## How to Run (Step by Step)

Run each file in order. Open your terminal in this folder and type:

**Step 1 – Clean the NSL-KDD data**
```
python Processing_script.py
```

**Step 2 – Select the best features using PSO, GA, and GWO**
```
python ensemble_features.py
```
⚠️ This step takes about 5–15 minutes. Just let it run.

**Step 3 – Train the models**
```
python classifiers.py
```

**Step 4 – Evaluate and generate graphs**
```
python step6_evaluate.py
```

**Step 5 – Clean the UNSW-NB15 data**
```
python step7_preprocess_NB15.py
```

**Step 6 – Feature selection on UNSW-NB15**
```
python step7_ensemble_NB15.py
```
⚠️ This also takes about 5–15 minutes.

**Step 7 – Train models on UNSW-NB15**
```
python step7_classifiers_NB15.py
```

**Step 8 – Evaluate UNSW-NB15 and generate graphs**
```
python step7_evaluate_NB15.py
```

---

## What Gets Generated

After running all steps, you will find these output files in the folder:

**Graphs (NSL-KDD)**
- `graph1_performance.png` – Bar chart comparing all models
- `graph2_roc.png` – ROC curves
- `graph3_time.png` – Training time comparison
- `graph4_confusion.png` – Confusion matrices

**Graphs (UNSW-NB15)**
- `NB15_graph1_performance.png`
- `NB15_graph2_roc.png`
- `NB15_graph3_time.png`
- `NB15_graph4_confusion.png`

**Metrics**
- `metrics.txt` – Full results table for NSL-KDD
- `NB15_metrics.txt` – Full results table for UNSW-NB15

---

## Key Results Summary

| Model | NSL-KDD Accuracy | UNSW-NB15 Accuracy |
|---|---|---|
| Baseline SVM (no optimization) | 77.39% | 93.87% |
| Optimised SVM (ensemble features) | 76.61% | 93.63% |
| **Random Forest (ensemble features)** | **78.53%** | **93.97%** |

The ensemble-optimised Random Forest is the best model on both datasets.
It also reduced false alarms on UNSW-NB15 from 19.14% down to 10.67%.

---

## The Three Algorithms (Simple Explanation)

- **PSO** – Particles fly around searching for the best features, like birds flocking
- **GA** – Features evolve over generations, survival of the fittest
- **GWO** – Wolves hunt together, led by the alpha, beta, and delta wolves

After all three run independently, a **majority vote** keeps only the features
that at least 2 out of 3 algorithms agreed on.

---

## Datasets Used

- **NSL-KDD** – Downloaded from https://www.unb.ca/cic/datasets/nsl.html
  - 20,000 samples used (randomly sampled, seed=42)
- **UNSW-NB15** – Downloaded from https://research.unsw.edu.au/projects/unsw-nb15-dataset
  - 15,000 samples used (randomly sampled, seed=42)

---

*CYS 417 – Machine Intelligence | Federal University of Technology, Akure*
