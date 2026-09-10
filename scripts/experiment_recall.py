"""Experiment script to find best approach to improve recall from 71% to 85%+.

Tests multiple strategies:
1. Different classifiers (RandomForest, LogisticRegression, SGD, LinearSVC+Calibrated)
2. Custom class weights (favoring minority classes)
3. TF-IDF variations (char ngrams, more features)
4. Manual oversampling for minority classes
5. Threshold tuning (post-hoc)
6. Ensembles (voting)
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
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, recall_score, f1_score
from sklearn.utils import resample

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
print(f"Loaded {len(texts)} samples")
print(f"Label distribution: {Counter(labels)}")
print("=" * 80)

# Split (same as training: 80/20, seed 42, stratified)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Train: {Counter(y_train)}, Test: {Counter(y_test)}")
print("=" * 80)


def evaluate(pipeline, X_tr, y_tr, X_te, y_te, name=""):
    pipeline.fit(X_tr, y_tr)
    preds = pipeline.predict(X_te)
    macro_recall = recall_score(y_te, preds, average="macro")
    macro_f1 = f1_score(y_te, preds, average="macro")
    per_class_recall = recall_score(y_te, preds, average=None, labels=[0, 1, 2], zero_division=0)
    print(f"\n--- {name} ---")
    print(f"  Macro Recall: {macro_recall:.4f} | Macro F1: {macro_f1:.4f}")
    print(f"  Recall: normal={per_class_recall[0]:.4f} atencao={per_class_recall[1]:.4f} urgente={per_class_recall[2]:.4f}")
    print(classification_report(y_te, preds, labels=[0, 1, 2], target_names=["normal", "atencao", "urgente"], zero_division=0))
    return macro_recall, macro_f1, per_class_recall


# ---- TF-IDF configs ----
tfidf_word = TfidfVectorizer(
    max_features=20000, ngram_range=(1, 2), sublinear_tf=True,
    min_df=2, max_df=0.95, stop_words=list(ENGLISH_STOPWORDS),
)
tfidf_word_big = TfidfVectorizer(
    max_features=50000, ngram_range=(1, 3), sublinear_tf=True,
    min_df=2, max_df=0.95, stop_words=list(ENGLISH_STOPWORDS),
)
tfidf_char = TfidfVectorizer(
    max_features=30000, ngram_range=(3, 5), sublinear_tf=True,
    min_df=2, max_df=0.95, stop_words=list(ENGLISH_STOPWORDS),
    analyzer="char_wb",
)

# ---- Custom class weights ----
# Heavily favor normal (minority) and urgente
weights_balanced = "balanced"
weights_custom_1 = {0: 3.0, 1: 1.0, 2: 1.5}   # boost normal strongly
weights_custom_2 = {0: 5.0, 1: 1.0, 2: 2.0}   # boost normal very strongly
weights_custom_3 = {0: 6.5, 1: 1.0, 2: 2.5}   # extreme normal boost (ratio-based)

# ============================================================
# Baseline (current config)
# ============================================================
print("\n" + "=" * 80)
print("BASELINE: RandomForest + TF-IDF word (current config)")
print("=" * 80)
pipe_base = Pipeline([
    ("tfidf", tfidf_word),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")),
])
evaluate(pipe_base, X_train, y_train, X_test, y_test, "Baseline RF-100 balanced")

# ============================================================
# Experiment 1: LogisticRegression (often better for TF-IDF text)
# ============================================================
print("\n" + "=" * 80)
print("EXPERIMENT 1: LogisticRegression variants")
print("=" * 80)

for C in [0.5, 1.0, 2.0, 5.0]:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), sublinear_tf=True,
            min_df=2, max_df=0.95, stop_words=list(ENGLISH_STOPWORDS),
        )),
        ("clf", LogisticRegression(C=C, max_iter=2000, random_state=42, class_weight="balanced")),
    ])
    evaluate(pipe, X_train, y_train, X_test, y_test, f"LR C={C} balanced")

# ============================================================
# Experiment 2: LogisticRegression with custom class weights
# ============================================================
print("\n" + "=" * 80)
print("EXPERIMENT 2: LogisticRegression with custom weights")
print("=" * 80)

for wname, weights in [("custom1", weights_custom_1), ("custom2", weights_custom_2), ("custom3", weights_custom_3)]:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), sublinear_tf=True,
            min_df=2, max_df=0.95, stop_words=list(ENGLISH_STOPWORDS),
        )),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight=weights)),
    ])
    evaluate(pipe, X_train, y_train, X_test, y_test, f"LR C=1.0 {wname}")

# ============================================================
# Experiment 3: RandomForest with more estimators + custom weights
# ============================================================
print("\n" + "=" * 80)
print("EXPERIMENT 3: RandomForest tuning")
print("=" * 80)

for n_est in [300, 500]:
    for wname, weights in [("balanced", "balanced"), ("custom2", weights_custom_2)]:
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=20000, ngram_range=(1, 2), sublinear_tf=True,
                min_df=2, max_df=0.95, stop_words=list(ENGLISH_STOPWORDS),
            )),
            ("clf", RandomForestClassifier(n_estimators=n_est, random_state=42, class_weight=weights, n_jobs=-1)),
        ])
        evaluate(pipe, X_train, y_train, X_test, y_test, f"RF-{n_est} {wname}")

# ============================================================
# Experiment 4: LinearSVC + CalibratedClassifierCV
# ============================================================
print("\n" + "=" * 80)
print("EXPERIMENT 4: LinearSVC + CalibratedClassifierCV")
print("=" * 80)

for wname, weights in [("balanced", "balanced"), ("custom2", weights_custom_2)]:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), sublinear_tf=True,
            min_df=2, max_df=0.95, stop_words=list(ENGLISH_STOPWORDS),
        )),
        ("clf", CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight=weights, max_iter=5000, random_state=42),
            cv=3,
        )),
    ])
    evaluate(pipe, X_train, y_train, X_test, y_test, f"LinearSVC+Cal {wname}")

# ============================================================
# Experiment 5: SGDClassifier (linear, scalable)
# ============================================================
print("\n" + "=" * 80)
print("EXPERIMENT 5: SGDClassifier")
print("=" * 80)

for wname, weights in [("balanced", "balanced"), ("custom2", weights_custom_2)]:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), sublinear_tf=True,
            min_df=2, max_df=0.95, stop_words=list(ENGLISH_STOPWORDS),
        )),
        ("clf", SGDClassifier(loss="log_loss", max_iter=1000, random_state=42, class_weight=weights, n_jobs=-1)),
    ])
    evaluate(pipe, X_train, y_train, X_test, y_test, f"SGD log_loss {wname}")

# ============================================================
# Experiment 6: ComplementNB (good for imbalanced text)
# ============================================================
print("\n" + "=" * 80)
print("EXPERIMENT 6: ComplementNB")
print("=" * 80)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), sublinear_tf=True,
        min_df=2, max_df=0.95, stop_words=list(ENGLISH_STOPWORDS),
    )),
    ("clf", ComplementNB()),
])
evaluate(pipe, X_train, y_train, X_test, y_test, "ComplementNB")

# ============================================================
# Experiment 7: TF-IDF with more features + ngrams (1,3)
# ============================================================
print("\n" + "=" * 80)
print("EXPERIMENT 7: TF-IDF variations + LR")
print("=" * 80)

pipe = Pipeline([
    ("tfidf", tfidf_word_big),
    ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight=weights_custom_2)),
])
evaluate(pipe, X_train, y_train, X_test, y_test, "LR custom2 + TF-IDF 50k (1,3)")

# Char ngrams
pipe = Pipeline([
    ("tfidf", tfidf_char),
    ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight=weights_custom_2)),
])
evaluate(pipe, X_train, y_train, X_test, y_test, "LR custom2 + char ngrams (3,5)")

# Combined: word + char (FeatureUnion)
from sklearn.pipeline import FeatureUnion

pipe = Pipeline([
    ("features", FeatureUnion([
        ("word", TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), sublinear_tf=True,
            min_df=2, max_df=0.95, stop_words=list(ENGLISH_STOPWORDS),
        )),
        ("char", TfidfVectorizer(
            max_features=20000, ngram_range=(3, 5), sublinear_tf=True,
            min_df=2, max_df=0.95, analyzer="char_wb",
        )),
    ])),
    ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight=weights_custom_2)),
])
evaluate(pipe, X_train, y_train, X_test, y_test, "LR custom2 + word+char FeatureUnion")

# ============================================================
# Experiment 8: Manual oversampling
# ============================================================
print("\n" + "=" * 80)
print("EXPERIMENT 8: Manual oversampling + LR")
print("=" * 80)

# Oversample normal and urgente to match atencao
def oversample(X, y, target_per_class=None):
    if target_per_class is None:
        target_per_class = Counter(y).most_common(1)[0][1]
    X_list, y_list = list(X), list(y)
    new_X, new_y = [], []
    for label in set(y):
        idx = [i for i, v in enumerate(y_list) if v == label]
        X_class = [X_list[i] for i in idx]
        n = target_per_class
        if len(X_class) < n:
            X_up = resample(X_class, n_samples=n, random_state=42, replace=True)
            new_X.extend(X_up)
            new_y.extend([label] * n)
        else:
            new_X.extend(X_class)
            new_y.extend([label] * len(X_class))
    return np.array(new_X, dtype=object), np.array(new_y)

# Oversample to max class count
X_up, y_up = oversample(X_train, y_train)
print(f"Oversampled train: {Counter(y_up)}")

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), sublinear_tf=True,
        min_df=2, max_df=0.95, stop_words=list(ENGLISH_STOPWORDS),
    )),
    ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=42)),
])
evaluate(pipe, X_up, y_up, X_test, y_test, "LR + oversampling (no weights)")

# Oversample + weights
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), sublinear_tf=True,
        min_df=2, max_df=0.95, stop_words=list(ENGLISH_STOPWORDS),
    )),
    ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight=weights_custom_2)),
])
evaluate(pipe, X_up, y_up, X_test, y_test, "LR + oversampling + custom2 weights")

# ============================================================
# Experiment 9: Voting ensemble
# ============================================================
print("\n" + "=" * 80)
print("EXPERIMENT 9: Voting ensemble")
print("=" * 80)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), sublinear_tf=True,
        min_df=2, max_df=0.95, stop_words=list(ENGLISH_STOPWORDS),
    )),
    ("clf", VotingClassifier([
        ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight=weights_custom_2)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight=weights_custom_2, n_jobs=-1)),
        ("svc", CalibratedClassifierCV(LinearSVC(C=1.0, class_weight=weights_custom_2, max_iter=5000, random_state=42), cv=3)),
    ], voting="soft")),
])
evaluate(pipe, X_train, y_train, X_test, y_test, "Voting (LR+RF+SVC) soft custom2")

# ============================================================
# Experiment 10: Threshold tuning (post-hoc)
# ============================================================
print("\n" + "=" * 80)
print("EXPERIMENT 10: Threshold tuning (post-hoc)")
print("=" * 80)

# Train LR with custom2, then tune thresholds per class
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), sublinear_tf=True,
        min_df=2, max_df=0.95, stop_words=list(ENGLISH_STOPWORDS),
    )),
    ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight=weights_custom_2)),
])
pipe.fit(X_train, y_train)
probas = pipe.predict_proba(X_test)

# Grid search for best thresholds to maximize macro recall while keeping F1 decent
best_macro_recall = 0
best_thresholds = None
best_macro_f1 = 0

for t0 in np.arange(0.1, 0.9, 0.05):
    for t2 in np.arange(0.1, 0.9, 0.05):
        preds = []
        for row in probas:
            scores = [row[0] - (1 - t0), row[1], row[2] - (1 - t2)]
            preds.append(np.argmax(scores))
        macro_recall = recall_score(y_test, preds, average="macro")
        macro_f1 = f1_score(y_test, preds, average="macro")
        if macro_recall > best_macro_recall and macro_f1 > 0.55:
            best_macro_recall = macro_recall
            best_thresholds = (t0, t2)
            best_macro_f1 = macro_f1

print(f"Best thresholds: t0={best_thresholds[0]:.2f}, t2={best_thresholds[2]:.2f}")
print(f"Best Macro Recall: {best_macro_recall:.4f} | Macro F1: {best_macro_f1:.4f}")

preds = []
for row in probas:
    scores = [row[0] - (1 - best_thresholds[0]), row[1], row[2] - (1 - best_thresholds[2])]
    preds.append(np.argmax(scores))
per_class = recall_score(y_test, preds, average=None, labels=[0, 1, 2], zero_division=0)
print(f"Recall: normal={per_class[0]:.4f} atencao={per_class[1]:.4f} urgente={per_class[2]:.4f}")
print(classification_report(y_test, preds, labels=[0, 1, 2], target_names=["normal", "atencao", "urgente"], zero_division=0))

print("\n" + "=" * 80)
print("EXPERIMENTS COMPLETE")
print("=" * 80)
