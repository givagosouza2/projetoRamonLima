# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier


# ----------------------------
# Helpers
# ----------------------------
def bootstrap_ci_mean(values: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 42):
    """Bootstrap CI for the mean of 'values' (e.g., CV fold scores)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, (np.nan, np.nan)

    n = values.size
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boots[i] = np.mean(sample)

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1.0 - alpha))
    return float(np.mean(values)), (lo, hi)


def clean_dataframe(df: pd.DataFrame):
    """
    Assumptions:
    - First column is target (categorical labels)
    - Remaining columns are numeric features
    """
    if df.shape[1] < 3:
        raise ValueError("O CSV parece ter poucas colunas. Esperado: 1 coluna de classe + >=2 features numéricas.")

    target_col = df.columns[0]
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Force numeric features
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    valid = (~X.isna().any(axis=1)) & (~y.isna())
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(str).reset_index(drop=True)

    return X, y, target_col


def make_models(random_state: int = 42):
    """Return dict of named estimators (pipelines). All include StandardScaler in a Pipeline."""
    models = {}

    models["LogReg"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                multi_class="multinomial",
                max_iter=3000,
                class_weight="balanced",
                random_state=random_state
            )),
        ]
    )

    models["LinearSVM"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="linear",
                probability=True,      # needed for soft voting
                class_weight="balanced",
                random_state=random_state
            )),
        ]
    )

    models["RandomForest"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=500,
                random_state=random_state,
                class_weight="balanced_subsample"
            )),
        ]
    )

    models["ExtraTrees"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(
                n_estimators=700,
                random_state=random_state,
                class_weight="balanced"
            )),
        ]
    )

    models["GradBoost"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=random_state)),
        ]
    )

    return models


def cv_evaluate(models: dict, X_train, y_train, n_splits: int, n_repeats: int, seed: int):
    """Cross-validate each model on training set only. Return per-model arrays of scores."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scoring = {"bal_acc": "balanced_accuracy", "macro_f1": "f1_macro"}

    out = {}
    for name, est in models.items():
        cvres = cross_validate(
            est, X_train, y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )
        out[name] = {
            "bal_acc": np.asarray(cvres["test_bal_acc"], dtype=float),
            "macro_f1": np.asarray(cvres["test_macro_f1"], dtype=float),
        }
    return out


def fit_and_test(est, X_train, y_train, X_test, y_test):
    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)
    bal = balanced_accuracy_score(y_test, y_pred)
    mf1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred, labels=sorted(pd.unique(y_test)))
    return bal, mf1, cm, y_pred


def build_weighted_ensemble(models: dict, cv_table: pd.DataFrame, weight_mode: str):
    """Build soft-voting ensemble; optionally weight by CV BalAcc mean."""
    estimators = [(name, models[name]) for name in models.keys()]

    if weight_mode == "CV Balanced Acc (ponderado)":
        weights = []
        for name in models.keys():
            w = float(cv_table.loc[cv_table["Modelo"] == name, "BalAcc_CV_mean"].iloc[0])
            weights.append(w)
        s = sum(weights)
        weights = [w / s if s > 0 else 1.0 for w in weights]
    else:
        weights = None

    ens = VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=weights,
        n_jobs=-1
    )
    return ens


def filter_pairwise(X: pd.DataFrame, y: pd.Series, class_a: str, class_b: str):
    """Keep only two selected classes."""
    mask = (y == class_a) | (y == class_b)
    X2 = X.loc[mask].reset_index(drop=True)
    y2 = y.loc[mask].reset_index(drop=True)
    return X2, y2


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Tremor ML - Classificação por idade", layout="wide")
st.title("Classificação por idade a partir de medidas de tremor (sensores inerciais)")
st.caption(
    "Pipeline: split 80/20 estratificado → CV repetida (só no treino) → bootstrap (IC) das métricas da CV → teste final.\n"
    "Normalização (StandardScaler) está sempre dentro do Pipeline para evitar vazamento."
)

with st.sidebar:
    st.header("Configurações gerais")
    test_size = st.slider("Tamanho do teste (proporção)", 0.10, 0.40, 0.20, 0.05)
    n_splits = st.slider("CV: número de folds", 3, 10, 5, 1)
    n_repeats = st.slider("CV: repetições", 1, 20, 10, 1)
    n_boot = st.slider("Bootstrap (IC): reamostragens", 200, 20000, 2000, 200)
    seed = st.number_input("Random seed", min_value=0, max_value=100_000, value=42, step=1)

    st.divider()
    st.subheader("Modo de análise")
    analysis_mode = st.radio("Escolha o modo", ["Multiclasse (todas as classes)", "Par-a-par (binário)"], index=0)

    st.divider()
    st.subheader("Ensemble (votação)")
    weight_mode = st.selectbox("Pesos do ensemble", ["CV Balanced Acc (ponderado)", "Uniforme"], index=0)

uploaded = st.file_uploader("Upload do CSV (1ª coluna = classe; demais = features numéricas)", type=["csv"])
use_local = st.checkbox("Usar arquivo local 'database.csv' (se existir no diretório do app)", value=False)

df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_local:
    try:
        df = pd.read_csv("database.csv")
    except Exception as e:
        st.error(f"Não consegui abrir 'database.csv' local: {e}")

if df is None:
    st.info("Envie um CSV para começar.")
    st.stop()

# Preview
with st.expander("Ver dados (primeiras linhas)"):
    st.dataframe(df.head(30), use_container_width=True)

# Clean
try:
    X, y, target_col = clean_dataframe(df)
except Exception as e:
    st.error(f"Erro ao preparar a base: {e}")
    st.stop()

st.write(f"**Coluna alvo (classe):** `{target_col}`")
st.write(f"**Amostras válidas:** {len(y)}  |  **Nº de features:** {X.shape[1]}")

# Distribution
dist = y.value_counts().rename_axis("classe").reset_index(name="n")
colA, colB = st.columns([1, 2])
with colA:
    st.markdown("### Distribuição de classes")
    st.dataframe(dist, use_container_width=True)
with colB:
    st.markdown("### Estatísticas das features")
    st.dataframe(X.describe().T, use_container_width=True)

# Pairwise selection
selected_pair = None
if analysis_mode == "Par-a-par (binário)":
    st.markdown("## Seleção do par de classes")
    classes = sorted(y.unique().tolist())
    if len(classes) < 2:
        st.error("Não há classes suficientes para análise par-a-par.")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        class_a = st.selectbox("Classe A", classes, index=0)
    with c2:
        class_b = st.selectbox("Classe B", [c for c in classes if c != class_a], index=0)

    selected_pair = (class_a, class_b)
    X, y = filter_pairwise(X, y, class_a, class_b)

    st.info(f"Rodando análise binária: **{class_a} × {class_b}**  |  Amostras usadas: **{len(y)}**")
    # update distribution after filtering
    dist2 = y.value_counts().rename_axis("classe").reset_index(name="n")
    st.dataframe(dist2, use_container_width=True)

# Train/test split (stratified)
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=float(test_size),
        stratify=y,
        random_state=int(seed),
    )
except Exception as e:
    st.error(f"Erro ao fazer split estratificado. Talvez alguma classe tenha poucos casos: {e}")
    st.stop()

st.success(f"Split concluído: treino={len(y_train)} | teste={len(y_test)} (estratificado)")

models = make_models(random_state=int(seed))

run_btn = st.button("Rodar avaliação", type="primary")
if not run_btn:
    st.stop()

# CV on training
with st.spinner("Rodando validação cruzada repetida no conjunto de treino..."):
    cv_scores = cv_evaluate(
        models, X_train, y_train,
        n_splits=int(n_splits),
        n_repeats=int(n_repeats),
        seed=int(seed)
    )

# Bootstrap CIs from CV fold scores
rows = []
for name, d in cv_scores.items():
    bal_mean, bal_ci = bootstrap_ci_mean(d["bal_acc"], n_boot=int(n_boot), ci=0.95, seed=int(seed))
    f1_mean, f1_ci = bootstrap_ci_mean(d["macro_f1"], n_boot=int(n_boot), ci=0.95, seed=int(seed) + 1)
    rows.append({
        "Modelo": name,
        "BalAcc_CV_mean": bal_mean,
        "BalAcc_CV_CI95": f"[{bal_ci[0]:.3f}, {bal_ci[1]:.3f}]",
        "MacroF1_CV_mean": f1_mean,
        "MacroF1_CV_CI95": f"[{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]",
    })

cv_table = pd.DataFrame(rows).sort_values(by="BalAcc_CV_mean", ascending=False).reset_index(drop=True)

st.subheader("Resultados no treino (CV repetida) + IC95% (bootstrap)")
st.dataframe(cv_table, use_container_width=True)

# Build ensemble AFTER seeing CV means (still training-only info)
ensemble = build_weighted_ensemble(models, cv_table, weight_mode=weight_mode)

# Fit/test each model on holdout test
test_rows = []
cms = {}
preds = {}

with st.spinner("Treinando no treino completo e avaliando no teste final..."):
    for name, est in models.items():
        bal, mf1, cm, y_pred = fit_and_test(est, X_train, y_train, X_test, y_test)
        test_rows.append({"Modelo": name, "BalAcc_Teste": bal, "MacroF1_Teste": mf1})
        cms[name] = cm
        preds[name] = y_pred

    bal_e, mf1_e, cm_e, y_pred_e = fit_and_test(ensemble, X_train, y_train, X_test, y_test)
    test_rows.append({"Modelo": "Ensemble(soft)", "BalAcc_Teste": bal_e, "MacroF1_Teste": mf1_e})
    cms["Ensemble(soft)"] = cm_e
    preds["Ensemble(soft)"] = y_pred_e

test_table = pd.DataFrame(test_rows).sort_values(by="BalAcc_Teste", ascending=False).reset_index(drop=True)

st.subheader("Resultados no teste final (holdout)")
st.dataframe(test_table, use_container_width=True)

# Choose a model to show details
choices = list(models.keys()) + ["Ensemble(soft)"]
default_idx = choices.index("Ensemble(soft)")
show_model = st.selectbox("Mostrar detalhes para o modelo:", choices, index=default_idx)

labels = sorted(pd.unique(y_test).tolist())
cm = cms[show_model]

st.markdown("### Matriz de confusão (teste)")
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, values_format="d")
st.pyplot(fig, clear_figure=True)

st.markdown("### Relatório de classificação (teste)")
rep = classification_report(y_test, preds[show_model], labels=labels, zero_division=0)
st.code(rep)

# Export tables
st.markdown("### Exportar resultados (CSV)")
out1 = io.StringIO()
cv_table.to_csv(out1, index=False)
cv_csv = out1.getvalue().encode("utf-8")

out2 = io.StringIO()
test_table.to_csv(out2, index=False)
test_csv = out2.getvalue().encode("utf-8")

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "Baixar tabela CV (CSV)",
        data=cv_csv,
        file_name="cv_results.csv",
        mime="text/csv",
    )
with col2:
    st.download_button(
        "Baixar tabela teste (CSV)",
        data=test_csv,
        file_name="test_results.csv",
        mime="text/csv",
    )

# Helpful caption
if analysis_mode == "Par-a-par (binário)" and selected_pair is not None:
    st.caption(
        f"Dica: compare os 3 pares (Jovem×Adulto, Jovem×Idoso, Adulto×Idoso). "
        f"Normalmente o par mais separável é o de extremos etários."
    )
else:
    st.caption(
        "Dica: a classe intermediária (Adulto) costuma ser a mais difícil. "
        "Use o modo Par-a-par para investigar separabilidade entre pares."
    )
