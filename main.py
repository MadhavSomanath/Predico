"""Train and honestly evaluate Predico's models, then save the best one.

Run: python main.py
"""
import json
import os

import numpy as np
import pandas as pd
import yaml
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
SEED = cfg["random_state"]


def load_data():
    paths = cfg["dataset"]
    df = pd.concat([pd.read_csv(paths["training_data_path"]), pd.read_csv(paths["test_data_path"])])
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]  # trailing comma in the CSV
    # 'fluid_overload' is all zeros; the real column is the duplicate 'fluid_overload.1'
    df = df.drop(columns="fluid_overload").rename(columns={"fluid_overload.1": "fluid_overload"})
    raw = len(df)
    # The 2020 split had 4,962 rows but only ~305 unique ones; 41/42 test rows were also in train.
    df = df.drop_duplicates()
    print(f"{raw} rows -> {len(df)} unique symptom patterns, {df.prognosis.nunique()} diseases")
    return df.drop(columns="prognosis"), df["prognosis"], raw


def models():
    m = cfg["model"]
    return {
        "decision_tree": DecisionTreeClassifier(criterion=m["decision_tree"]["criterion"], random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=m["random_forest"]["n_estimators"], random_state=SEED),
        "naive_bayes": BernoulliNB(alpha=m["naive_bayes"]["alpha"]),
    }


def keep_k_symptoms(X, k, rng):
    """Simulate a real user who only reports k of the patient's symptoms."""
    out = np.zeros_like(X)
    for i, row in enumerate(X):
        on = np.flatnonzero(row)
        out[i, rng.choice(on, min(k, len(on)), replace=False)] = 1
    return out


def topk_hits(clf, X, y, k):
    proba = clf.predict_proba(X)
    top = clf.classes_[np.argsort(proba, axis=1)[:, ::-1][:, :k]]
    return (top == np.asarray(y)[:, None]).any(axis=1)


def evaluate(X, y):
    """5-fold CV on deduplicated data, scored on full and partial (3-symptom) inputs."""
    k_partial = cfg["eval"]["partial_symptoms"]
    rng = np.random.default_rng(SEED)
    folds = StratifiedKFold(n_splits=cfg["eval"]["folds"], shuffle=True, random_state=SEED)
    scores = {name: {"top1_full": [], "top1_partial": [], "top3_partial": []} for name in models()}
    Xv, yv = X.to_numpy(), y.to_numpy()
    for tr, te in folds.split(Xv, yv):
        Xp = keep_k_symptoms(Xv[te], k_partial, rng)
        for name, clf in models().items():
            clf.fit(Xv[tr], yv[tr])
            scores[name]["top1_full"] += list(topk_hits(clf, Xv[te], yv[te], 1))
            scores[name]["top1_partial"] += list(topk_hits(clf, Xp, yv[te], 1))
            scores[name]["top3_partial"] += list(topk_hits(clf, Xp, yv[te], 3))
    return {n: {m: round(float(np.mean(v)), 3) for m, v in s.items()} for n, s in scores.items()}


if __name__ == "__main__":
    X, y, raw_rows = load_data()
    results = evaluate(X, y)
    print(pd.DataFrame(results).T.to_string())

    best = max(results, key=lambda n: results[n]["top3_partial"])
    print(f"\nBest on realistic input (top-3 with {cfg['eval']['partial_symptoms']} symptoms): {best}")
    clf = models()[best].fit(X, y)
    os.makedirs(cfg["model_save_path"], exist_ok=True)
    dump(clf, cfg["model_save_path"] + "model.joblib", compress=3)
    with open(cfg["model_save_path"] + "metrics.json", "w") as f:
        json.dump({"best": best, "raw_rows": raw_rows, "unique_rows": len(X),
                   "diseases": int(y.nunique()), "partial_symptoms": cfg["eval"]["partial_symptoms"],
                   "results": results}, f, indent=2)
