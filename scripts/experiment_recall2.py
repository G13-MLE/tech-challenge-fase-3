"""Phase 2: Optimized threshold search to push recall to 85%+.

Uses numpy vectorization for fast threshold grid search.
"""

import csv
import sys
import warnings
from collections import Counter
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, recall_score, f1_score, precision_score

from src.core.stopwords import ENGLISH_STOPWORDS

warnings.filterwarnings("ignore")

# Load data
texts, labels = [], []
with open("data/processed/laudos_processed.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        texts.append(row["text"])
        labels.append(int(row["label"]))

texts = np.array(texts, dtype=object)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Train: {Counter(y_train)}, Test: {Counter(y_test)}")
print("=" * 80)

STOPWORDS = list(ENGLISH_STOPWORDS)


def make_tfidf():
    return TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), sublinear_tf=True,
        min_df=2, max_df=0.95, stop_words=STOPWORDS,
    )


def evaluate_preds(y_te, preds, name=""):
    macro_recall = recall_score(y_te, preds, average="macro")
    macro_f1 = f1_score(y_te, preds, average="macro")
    macro_prec = precision_score(y_te, preds, average="macro")
    per_class = recall_score(y_te, preds, average=None, labels=[0, 1, 2], zero_division=0)
    print(f"\n--- {name} ---")
    print(f"  Macro Recall: {macro_recall:.4f} | Macro F1: {macro_f1:.4f} | Macro Prec: {macro_prec:.4f}")
    print(f"  Recall: normal={per_class[0]:.4f} atencao={per_class[1]:.4f} urgente={per_class[2]:.4f}")
    print(classification_report(y_te, preds, labels=[0, 1, 2], target_names=["normal", "atencao", "urgente"], zero_division=0))
    return macro_recall, macro_f1, per_class


# Train best model: LR C=0.5 balanced
print("\nTraining LR C=0.5 balanced...")
pipe = Pipeline([
    ("tfidf", make_tfidf()),
    ("clf", LogisticRegression(C=0.5, max_iter=2000, random_state=42, class_weight="balanced")),
])
pipe.fit(X_train, y_train)
probas_test = pipe.predict_proba(X_test)
preds_default = pipe.predict(X_test)
evaluate_preds(y_test, preds_default, "LR C=0.5 balanced (default)")

# Get CV probabilities for threshold tuning
print("\nComputing CV predictions for threshold tuning...")
cv_probas = cross_val_predict(pipe, X_train, y_train, cv=5, method="predict_proba")

# ============================================================
# Vectorized 2-way threshold search (t0, t2)
# ============================================================
print("\n" + "=" * 80)
print("VECTORIZED 2-WAY THRESHOLD SEARCH")
print("=" * 80)

# Create threshold grid
t_range = np.arange(-0.5, 1.5, 0.025)
n_samples = cv_probas.shape[0]

# For efficiency, precompute argmax for each (t0, t2) combination
best_score = -1
best_config = None

p0 = cv_probas[:, 0]  # normal probs
p1 = cv_probas[:, 1]  # atencao probs
p2 = cv_probas[:, 2]  # urgente probs

print(f"Grid size: {len(t_range)}x{len(t_range)} = {len(t_range)**2} combinations")
print(f"Samples: {n_samples}")

for t0 in t_range:
    for t2 in t_range:
        # Adjusted scores
        s0 = p0 + t0
        s1 = p1
        s2 = p2 + t2
        preds = np.argmax(np.column_stack([s0, s1, s2]), axis=1)
        macro_recall = recall_score(y_train, preds, average="macro")
        macro_f1 = f1_score(y_train, preds, average="macro")
        if macro_recall > best_score and macro_f1 > 0.55:
            best_score = macro_recall
            best_config = (t0, t2)

t0_best, t2_best = best_config
print(f"\nBest CV thresholds: t0={t0_best:.3f}, t2={t2_best:.3f}")
print(f"Best CV macro recall: {best_score:.4f}")

# Apply to test
s0 = probas_test[:, 0] + t0_best
s1 = probas_test[:, 1]
s2 = probas_test[:, 2] + t2_best
preds_test = np.argmax(np.column_stack([s0, s1, s2]), axis=1)
evaluate_preds(y_test, preds_test, f"LR C=0.5 + 2-way threshold (t0={t0_best:.3f}, t2={t2_best:.3f})")

# ============================================================
# 3-way threshold search (coarser grid)
# ============================================================
print("\n" + "=" * 80)
print("3-WAY THRESHOLD SEARCH (coarser)")
print("=" * 80)

t_range_coarse = np.arange(-0.5, 1.5, 0.1)
best_score = -1
best_config = None

print(f"Grid size: {len(t_range_coarse)}^3 = {len(t_range_coarse)**3} combinations")

for t0 in t_range_coarse:
    for t1 in t_range_coarse:
        for t2 in t_range_coarse:
            s0 = p0 + t0
            s1 = p1 + t1
            s2 = p2 + t2
            preds = np.argmax(np.column_stack([s0, s1, s2]), axis=1)
            macro_recall = recall_score(y_train, preds, average="macro")
            macro_f1 = f1_score(y_train, preds, average="macro")
            if macro_recall > best_score and macro_f1 > 0.55:
                best_score = macro_recall
                best_config = (t0, t1, t2)

t0_best, t1_best, t2_best = best_config
print(f"\nBest 3-way CV thresholds: t0={t0_best:.3f}, t1={t1_best:.3f}, t2={t2_best:.3f}")
print(f"Best CV macro recall: {best_score:.4f}")

s0 = probas_test[:, 0] + t0_best
s1 = probas_test[:, 1] + t1_best
s2 = probas_test[:, 2] + t2_best
preds_test = np.argmax(np.column_stack([s0, s1, s2]), axis=1)
evaluate_preds(y_test, preds_test, f"LR C=0.5 + 3-way threshold (t0={t0_best:.3f}, t1={t1_best:.3f}, t2={t2_best:.3f})")

# ============================================================
# Targeted: maximize recall_urgente specifically
# ============================================================
print("\n" + "=" * 80)
print("TARGETED: maximize recall_urgente >= 85%")
print("=" * 80)

best_urgente = -1
best_config = None

for t0 in np.arange(-0.5, 1.5, 0.025):
    for t2 in np.arange(-0.5, 2.0, 0.025):
        s0 = p0 + t0
        s2 = p2 + t2
        s1 = p1
        preds = np.argmax(np.column_stack([s0, s1, s2]), axis=1)
        per_class = recall_score(y_train, preds, average=None, labels=[0, 1, 2], zero_division=0)
        macro_f1 = f1_score(y_train, preds, average="macro")
        if per_class[2] >= 0.85 and macro_f1 > 0.55:
            macro_recall = recall_score(y_train, preds, average="macro")
            if macro_recall > best_urgente:
                best_urgente = macro_recall
                best_config = (t0, t2)

if best_config:
    t0_best, t2_best = best_config
    print(f"Best targeted thresholds: t0={t0_best:.3f}, t2={t2_best:.3f}")
    s0 = probas_test[:, 0] + t0_best
    s2 = probas_test[:, 2] + t2_best
    s1 = probas_test[:, 1]
    preds_test = np.argmax(np.column_stack([s0, s1, s2]), axis=1)
    evaluate_preds(y_test, preds_test, f"LR C=0.5 targeted urgente (t0={t0_best:.3f}, t2={t2_best:.3f})")
else:
    print("No config met urgente>=85% with F1>0.55. Trying F1>0.45...")
    best_urgente = -1
    best_config = None
    for t0 in np.arange(-0.5, 1.5, 0.025):
        for t2 in np.arange(-0.5, 2.0, 0.025):
            s0 = p0 + t0
            s2 = p2 + t2
            s1 = p1
            preds = np.argmax(np.column_stack([s0, s1, s2]), axis=1)
            per_class = recall_score(y_train, preds, average=None, labels=[0, 1, 2], zero_division=0)
            macro_f1 = f1_score(y_train, preds, average="macro")
            if per_class[2] >= 0.85 and macro_f1 > 0.45:
                macro_recall = recall_score(y_train, preds, average="macro")
                if macro_recall > best_urgente:
                    best_urgente = macro_recall
                    best_config = (t0, t2)
    if best_config:
        t0_best, t2_best = best_config
        print(f"Best (relaxed) thresholds: t0={t0_best:.3f}, t2={t2_best:.3f}")
        s0 = probas_test[:, 0] + t0_best
        s2 = probas_test[:, 2] + t2_best
        s1 = probas_test[:, 1]
        preds_test = np.argmax(np.column_stack([s0, s1, s2]), axis=1)
        evaluate_preds(y_test, preds_test, f"LR C=0.5 relaxed (t0={t0_best:.3f}, t2={t2_best:.3f})")

# ============================================================
# Oversampling + threshold tuning
# ============================================================
print("\n" + "=" * 80)
print("OVERSAMPLING + THRESHOLD TUNING")
print("=" * 80)

from sklearn.utils import resample

def oversample(X, y):
    target = Counter(y).most_common(1)[0][1]
    X_list, y_list = list(X), list(y)
    new_X, new_y = [], []
    for label in set(y):
        idx = [i for i, v in enumerate(y_list) if v == label]
        X_class = [X_list[i] for i in idx]
        if len(X_class) < target:
            X_up = resample(X_class, n_samples=target, random_state=42, replace=True)
            new_X.extend(X_up)
            new_y.extend([label] * target)
        else:
            new_X.extend(X_class)
            new_y.extend([label] * len(X_class))
    return np.array(new_X, dtype=object), np.array(new_y)

X_up, y_up = oversample(X_train, y_train)
print(f"Oversampled: {Counter(y_up)}")

pipe_os = Pipeline([
    ("tfidf", make_tfidf()),
    ("clf", LogisticRegression(C=0.5, max_iter=2000, random_state=42, class_weight="balanced")),
])
pipe_os.fit(X_up, y_up)
probas_os_test = pipe_os.predict_proba(X_test)
preds_os = pipe_os.predict(X_test)
evaluate_preds(y_test, preds_os, "OS+LR C=0.5 balanced (default)")

# Threshold tuning on oversampled
cv_probas_os = cross_val_predict(pipe_os, X_up, y_up, cv=5, method="predict_proba")
p0_os = cv_probas_os[:, 0]
p1_os = cv_probas_os[:, 1]
p2_os = cv_probas_os[:, 2]

best_score = -1
best_config = None
for t0 in np.arange(-0.5, 1.5, 0.025):
    for t2 in np.arange(-0.5, 1.5, 0.025):
        s0 = p0_os + t0
        s1 = p1_os
        s2 = p2_os + t2
        preds = np.argmax(np.column_stack([s0, s1, s2]), axis=1)
        macro_recall = recall_score(y_up, preds, average="macro")
        macro_f1 = f1_score(y_up, preds, average="macro")
        if macro_recall > best_score and macro_f1 > 0.55:
            best_score = macro_recall
            best_config = (t0, t2)

t0_best, t2_best = best_config
print(f"\nBest OS CV thresholds: t0={t0_best:.3f}, t2={t2_best:.3f}")
s0 = probas_os_test[:, 0] + t0_best
s1 = probas_os_test[:, 1]
s2 = probas_os_test[:, 2] + t2_best
preds_test = np.argmax(np.column_stack([s0, s1, s2]), axis=1)
evaluate_preds(y_test, preds_test, f"OS+LR C=0.5 + threshold (t0={t0_best:.3f}, t2={t2_best:.3f})")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("Baseline (RF-100 balanced):       Macro Recall=0.6561, urgente=0.7109")
print("Best model (LR C=0.5 balanced):    Macro Recall=0.7819, urgente=0.7962")
print(f"With threshold tuning:             Macro Recall={best_score:.4f} (CV)")
print("Target: Macro Recall >= 0.85")
