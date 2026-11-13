"""Predico: symptom-based disease predictor (Streamlit app). Run: streamlit run app.py"""
import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Predico", page_icon="🩺", layout="centered")


@st.cache_resource
def load():
    with open("saved_model/metrics.json") as f:
        return joblib.load("saved_model/model.joblib"), json.load(f)


model, metrics = load()
features = list(model.feature_names_in_)
label = {f: " ".join(f.replace("_", " ").split()).capitalize() for f in features}
pct = lambda m, k: f"{metrics['results'][metrics['best']][k]:.0%}"

st.title("🩺 Predico")
st.caption("A smart health predictor, first built in 2020 and rebuilt in 2026.")

check, retro = st.tabs(["Check symptoms", "2020 → 2026"])

with check:
    st.warning(
        "**This is a learning project, not medical advice.** It was trained on a small public dataset "
        "of 41 diseases. If you feel unwell, see a doctor. In an emergency, call your local emergency number.",
        icon="⚠️",
    )
    chosen = st.multiselect(
        "What symptoms do you have?", features, format_func=label.get,
        placeholder="Start typing, e.g. headache, fever, cough",
    )
    if st.button("Check", type="primary", disabled=not chosen):
        query = pd.DataFrame([[int(f in chosen) for f in features]], columns=features)
        proba = pd.Series(model.predict_proba(query)[0], index=model.classes_).nlargest(5)
        st.subheader("Most likely matches")
        st.dataframe(
            proba.rename("Model confidence").rename_axis("Condition").reset_index(),
            hide_index=True, width="stretch",
            column_config={"Model confidence": st.column_config.ProgressColumn(
                format="percent", min_value=0.0, max_value=1.0)},
        )
        if len(chosen) < metrics["partial_symptoms"]:
            st.info(f"With fewer than {metrics['partial_symptoms']} symptoms, results are much less reliable. "
                    "Add more if you can.")
        st.caption(
            f"Confidence is the model's vote share, not a medical probability. In testing on unseen cases "
            f"with only {metrics['partial_symptoms']} symptoms, the right condition was the top match "
            f"{pct(metrics, 'top1_partial')} of the time and in the top 3 {pct(metrics, 'top3_partial')} of the time."
        )

with retro:
    st.markdown(f"""
### What changed since 2020

| | 2020 | 2026 |
|---|---|---|
| **Data** | {metrics['raw_rows']:,} rows, used as-is | {metrics['unique_rows']} unique cases after removing duplicates |
| **Testing** | 41 of 42 test cases were copies of training cases | 5-fold cross-validation on unique cases only |
| **Headline score** | 98%, but tested on copies of training data | {pct(metrics, 'top1_partial')} top match / {pct(metrics, 'top3_partial')} top 3, from {metrics['partial_symptoms']} symptoms |
| **Output** | One diagnosis, shown in red | Top 5 matches with confidence |
| **Safety** | "Consult your nearest health administrator" | Clear disclaimer and a low-input warning |
| **Shipping** | Heroku "coming soon", a 137-package `pip freeze` | Streamlit Community Cloud, 6 pinned packages |

### Honest scores (unseen cases)
""")
    names = {"decision_tree": "Decision tree", "random_forest": "Random forest", "naive_bayes": "Naive Bayes"}
    st.dataframe(
        pd.DataFrame(metrics["results"]).T.rename(index=names).rename(columns={
            "top1_full": "Top match, all symptoms",
            "top1_partial": f"Top match, {metrics['partial_symptoms']} symptoms",
            "top3_partial": f"In top 3, {metrics['partial_symptoms']} symptoms",
        }),
        width="stretch",
        column_config={c: st.column_config.NumberColumn(format="percent") for c in [
            "Top match, all symptoms", f"Top match, {metrics['partial_symptoms']} symptoms",
            f"In top 3, {metrics['partial_symptoms']} symptoms"]},
    )
    st.markdown("""
### Lessons learned
- **Data quality beats model choice.** The 2020 test (98%) used copies of the training data, so it proved nothing.
  Real users give 2 or 3 symptoms, not 17, so that's what we test on now.
- **Ship it, and make it reproducible.** Working training script, pinned dependencies, one-click deploy.
- **Health AI must be humble.** Show uncertainty, show alternatives, and never pose as a doctor.
- **LLMs changed the field.** In 2026, symptom checkers let you describe symptoms in your own words.
  Predico stays classic ML on purpose: small, explainable, free to run and no API key needed.
""")
