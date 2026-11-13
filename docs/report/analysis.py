"""Analyses behind the 2026 report: figures + numbers.

Run from the repo root:  pip install matplotlib && python docs/report/analysis.py
Writes docs/report/figures/*.svg and docs/report/results.json.
"""
import json
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, ".")
from main import SEED, keep_k_symptoms, load_data, models  # noqa: E402

OUT = "docs/report"
REPEATS = 3  # CV repeats with fresh symptom draws, to smooth the 3-symptom sampling noise
KS = [1, 2, 3, 4, 5, 6]
INK, ACCENT, MUTED, WARN = "#14211d", "#0b6e63", "#8a9a95", "#a2481b"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})


def cv_predictions(X, y):
    """Out-of-fold probabilities for every model, k and repeat. Returns {(model, k): [(y_true, proba, classes)]}."""
    out = {}
    Xv, yv = X.to_numpy(), y.to_numpy()
    for r in range(REPEATS):
        rng = np.random.default_rng(SEED + r)
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + r)
        for tr, te in folds.split(Xv, yv):
            fitted = {n: m.fit(Xv[tr], yv[tr]) for n, m in models().items()}
            for k in KS + ["all"]:
                Xt = Xv[te] if k == "all" else keep_k_symptoms(Xv[te], k, rng)
                for n, clf in fitted.items():
                    out.setdefault((n, k), []).append((yv[te], clf.predict_proba(Xt), clf.classes_))
    return out


def stack(chunks):
    y = np.concatenate([c[0] for c in chunks])
    p = np.concatenate([c[1] for c in chunks])
    return y, p, chunks[0][2]


def topk(y, p, classes, k):
    top = classes[np.argsort(p, axis=1)[:, ::-1][:, :k]]
    return float((top == y[:, None]).any(axis=1).mean())


def save(fig, name):
    fig.savefig(f"{OUT}/figures/{name}.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(f"{OUT}/figures", exist_ok=True)
    X, y, raw_rows = load_data()
    res = {}

    # --- Data audit
    per_case = X.sum(axis=1)
    diseases_per_symptom = (X.groupby(y).max() > 0).sum(axis=0)
    res["data"] = {
        "raw_rows": raw_rows, "unique_rows": len(X), "diseases": int(y.nunique()), "symptoms": X.shape[1],
        "cases_per_disease_min": int(y.value_counts().min()), "cases_per_disease_max": int(y.value_counts().max()),
        "symptoms_per_case_min": int(per_case.min()), "symptoms_per_case_median": float(per_case.median()),
        "symptoms_per_case_max": int(per_case.max()),
        "symptoms_unused": int((X.sum(axis=0) == 0).sum()),
        "symptoms_in_one_disease": int((diseases_per_symptom == 1).sum()),
        "symptom_max_diseases": [diseases_per_symptom.idxmax(), int(diseases_per_symptom.max())],
    }

    preds = cv_predictions(X, y)

    # --- Symptom-count curve
    names = {"decision_tree": "Decision tree", "random_forest": "Random forest", "naive_bayes": "Naive Bayes"}
    curve = {n: {str(k): {"top1": topk(*stack(preds[(n, k)]), 1), "top3": topk(*stack(preds[(n, k)]), 3)}
                 for k in KS + ["all"]} for n in names}
    res["curve"] = curve
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for n, c, ls in [("random_forest", ACCENT, "-"), ("naive_bayes", INK, "-"), ("decision_tree", MUTED, "-")]:
        ax.plot(KS, [curve[n][str(k)]["top1"] for k in KS], ls, color=c, marker="o", ms=4, label=f"{names[n]}, top match")
    ax.plot(KS, [curve["random_forest"][str(k)]["top3"] for k in KS], "--", color=ACCENT, marker="o", ms=4,
            label="Random forest, in top 3")
    ax.set_xlabel("Symptoms the user reports"); ax.set_ylabel("Correct, unseen cases")
    ax.set_ylim(0, 1.02); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xticks(KS); ax.grid(axis="y", color="#e3e9e7", lw=0.6); ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(fig, "curve")

    # --- Error analysis (random forest, 3 symptoms)
    yt, p, cl = stack(preds[("random_forest", 3)])
    pred = cl[p.argmax(axis=1)]
    wrong = Counter((t, q) for t, q in zip(yt, pred) if t != q)
    totals = Counter(yt)
    res["confusions"] = [{"true": t, "predicted": q, "count": c, "rate": round(c / totals[t], 3)}
                         for (t, q), c in wrong.most_common(10)]
    recall = {d: float(np.mean(pred[yt == d] == d)) for d in sorted(totals)}
    res["recall_by_disease"] = dict(sorted(recall.items(), key=lambda kv: kv[1]))
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    items = sorted(recall.items(), key=lambda kv: kv[1])
    ax.barh([d for d, _ in items], [v for _, v in items],
            color=[WARN if v < 0.5 else ACCENT for _, v in items], height=0.7)
    ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Top match correct, 3 symptoms"); ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", color="#e3e9e7", lw=0.6)
    save(fig, "recall")

    # --- Calibration (3 symptoms)
    bins = np.linspace(0, 1, 11)
    res["calibration"] = {}
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":", label="Perfect calibration")
    for n, c in [("random_forest", ACCENT), ("naive_bayes", INK)]:
        yt, p, cl = stack(preds[(n, 3)])
        conf, ok = p.max(axis=1), cl[p.argmax(axis=1)] == yt
        idx = np.clip(np.digitize(conf, bins) - 1, 0, 9)
        rows = [(conf[idx == b].mean(), ok[idx == b].mean(), int((idx == b).sum())) for b in range(10) if (idx == b).any()]
        ece = sum(nb * abs(a - c_) for c_, a, nb in rows) / len(conf)
        res["calibration"][n] = {"ece": round(float(ece), 3), "mean_confidence": round(float(conf.mean()), 3),
                                 "accuracy": round(float(ok.mean()), 3),
                                 "share_below_30pct": round(float((conf < 0.3).mean()), 3),
                                 "bins": [[round(float(a), 3), round(float(b), 3), nb] for a, b, nb in rows]}
        ax.plot([r[0] for r in rows], [r[1] for r in rows], color=c, marker="o", ms=4, label=names[n])
    ax.set_xlabel("Model confidence (top match)"); ax.set_ylabel("Actually correct")
    for f in (ax.xaxis, ax.yaxis): f.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(color="#e3e9e7", lw=0.6); ax.legend(frameon=False, fontsize=8)
    save(fig, "calibration")

    # --- Explainability: global importance + signature symptoms per disease
    rf = models()["random_forest"].fit(X, y)
    imp = sorted(zip(X.columns, rf.feature_importances_), key=lambda kv: -kv[1])[:15]
    res["importance_top15"] = [[s, round(float(v), 4)] for s, v in imp]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    lab = [" ".join(s.replace("_", " ").split()) for s, _ in imp][::-1]
    ax.barh(lab, [v for _, v in imp][::-1], color=ACCENT, height=0.7)
    ax.set_xlabel("Random forest importance (mean decrease in impurity)"); ax.grid(axis="x", color="#e3e9e7", lw=0.6)
    save(fig, "importance")
    # P(disease | symptom) for symptoms present in >= 50% of that disease's cases: how much one symptom gives it away
    p_s_given_d = X.groupby(y).mean()
    p_d_given_s = X.groupby(y).sum() / X.sum(axis=0).replace(0, np.nan)
    sig = {}
    for d in p_s_given_d.index:
        common = p_s_given_d.loc[d][p_s_given_d.loc[d] >= 0.5].index
        best = p_d_given_s.loc[d, common].sort_values(ascending=False).head(3)
        sig[d] = [[s, round(float(v), 2)] for s, v in best.items()]
    res["signature_symptoms"] = sig

    with open(f"{OUT}/results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: res[k] for k in ("data", "calibration")}, indent=1))
    print(json.dumps(res["confusions"], indent=1))
    print({n: {k: (round(v["top1"], 3), round(v["top3"], 3)) for k, v in c.items()} for n, c in curve.items()})
    print(list(res["recall_by_disease"].items())[:8], sum(v >= 0.95 for v in recall.values()))
    print(res["importance_top15"][:8])
