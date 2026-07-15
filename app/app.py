from __future__ import annotations
import sys, secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from DataPipeline.config import ABT_PATH, ID_COL, TARGET_COL, RISK_THRESHOLD, MINIO_BUCKET
from MLOps import storage
from Model.predict import score_dataframe

st.set_page_config(
    page_title="Crédito para Pequenos Empresários",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.stApp {background:#f6f8fc; color:#0f172a;}
.block-container {padding-top:1.2rem; padding-bottom:3rem;}
[data-testid="stSidebar"] {background:#0f172a;}
[data-testid="stSidebar"] * {color:#f8fafc;}
.hero {
    padding:1.5rem 1.7rem;
    border-radius:18px;
    background:linear-gradient(120deg,#0f172a 0%,#1e3a8a 55%,#2563eb 100%);
    color:white;
    box-shadow:0 12px 35px rgba(30,58,138,.20);
    margin-bottom:1rem;
}
.hero h1 {margin:0 0 .35rem; font-size:2rem; color:#ffffff;}
.hero p {margin:0; opacity:.95; color:#e2e8f0;}
.soft-card {
    background:#ffffff;
    border:1px solid #e5e7eb;
    border-radius:14px;
    padding:1rem 1.15rem;
    box-shadow:0 5px 16px rgba(15,23,42,.05);
    min-height:110px;
}
.soft-label {font-size:.76rem; color:#475569; text-transform:uppercase; letter-spacing:.05em; font-weight:700;}
.soft-value {font-size:1.3rem; font-weight:750; color:#0f172a; margin-top:.25rem;}
.soft-note {font-size:.78rem; color:#64748b; margin-top:.2rem;}
.decision-approved, .decision-denied, .decision-review {
    padding:1.25rem 1.4rem;
    border-radius:16px;
    margin:.4rem 0 1rem;
    box-shadow:0 8px 22px rgba(15,23,42,.10);
}
.decision-approved {background:#ecfdf5; border:1px solid #86efac; color:#14532d;}
.decision-denied {background:#fef2f2; border:1px solid #fca5a5; color:#7f1d1d;}
.decision-review {background:#fff7ed; border:1px solid #fdba74; color:#7c2d12;}
.decision-title {font-size:1.35rem; font-weight:800; margin-bottom:.3rem;}
.threshold-note {
    background:rgba(255,255,255,.08);
    border:1px solid rgba(255,255,255,.16);
    padding:.8rem;
    border-radius:12px;
    font-size:.86rem;
}
.stButton > button, .stDownloadButton > button {
    border-radius:10px;
    font-weight:700;
    min-height:2.8rem;
}
/* Improve metric contrast */
div[data-testid="stMetric"] {
    background:#ffffff;
    border:1px solid #e5e7eb;
    padding:.85rem 1rem;
    border-radius:14px;
    box-shadow:0 4px 14px rgba(15,23,42,.04);
}
[data-testid="stMetricLabel"] {
    color:#475569 !important;
    font-weight:700 !important;
}
[data-testid="stMetricValue"] {
    color:#0f172a !important;
    font-weight:800 !important;
}
[data-testid="stMetricDelta"] {
    color:#1d4ed8 !important;
    font-weight:700 !important;
}
/* Improve tabs contrast */
.stTabs [data-baseweb="tab-list"] {gap:8px;}
.stTabs [data-baseweb="tab"] {
    background:#e2e8f0;
    border-radius:10px 10px 0 0;
    color:#0f172a;
    font-weight:700;
    padding:.55rem .85rem;
}
.stTabs [aria-selected="true"] {
    background:#ffffff !important;
    color:#dc2626 !important;
    border-bottom:2px solid #dc2626;
}
/* Improve alert readability */
div[data-testid="stAlert"] p, div[data-testid="stAlert"] li, div[data-testid="stAlert"] code {
    color:#0f172a !important;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=30, show_spinner=False)
def load_csv(key):
    return storage.read_csv(key, low_memory=False) if storage.exists(key) else None


@st.cache_data(ttl=30, show_spinner=False)
def load_json(key):
    return storage.read_json(key) if storage.exists(key) else None


def fmt_pct(v):
    return "-" if v is None else f"{float(v):.2%}"


def fmt_num(v, n=4):
    return "-" if v is None else f"{float(v):.{n}f}"


def money(v):
    if v is None or pd.isna(v):
        return "-"
    return f"R$ {float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def card(label, value, note=""):
    st.markdown(
        f'<div class="soft-card"><div class="soft-label">{label}</div><div class="soft-value">{value}</div><div class="soft-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def gauge(pd_value, threshold):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pd_value,
            number={"valueformat": ".3f", "font": {"size": 46, "color": "#0f172a"}},
            title={"text": "Probabilidade de default (PD)", "font": {"size": 18, "color": "#0f172a"}},
            gauge={
                "axis": {"range": [0, 1], "tickformat": ".1f", "tickcolor": "#475569"},
                "bar": {"color": "#2563eb"},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#cbd5e1",
                "steps": [
                    {"range": [0, threshold], "color": "#dcfce7"},
                    {"range": [threshold, 1], "color": "#fee2e2"},
                ],
                "threshold": {"line": {"color": "#111827", "width": 5}, "thickness": .82, "value": threshold},
            },
        )
    )
    fig.update_layout(height=370, margin=dict(l=35, r=35, t=70, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def show_decision(row, threshold, source_label):
    scored = score_dataframe(row, threshold=threshold)
    pdv = float(scored.PD_DEFAULT.iloc[0])
    decision = str(scored.CREDIT_DECISION.iloc[0])
    suggestion = str(scored.ACTION_SUGGESTION.iloc[0])
    approved = decision == "CONCEDER"

    left, right = st.columns([1.25, 1])
    with left:
        st.plotly_chart(gauge(pdv, threshold), use_container_width=True)
    with right:
        css = "decision-approved" if approved else "decision-denied"
        icon = "✅" if approved else "⛔"
        title = "EMPRÉSTIMO PODE SER CONCEDIDO" if approved else "REVISAR / NÃO CONCEDER AUTOMATICAMENTE"
        cid = row[ID_COL].iloc[0] if ID_COL in row else "simulação"
        st.markdown(
            f'<div class="{css}"><div class="decision-title">{icon} {title}</div><div><b>Origem:</b> {source_label}</div><div><b>Solicitação:</b> {cid}</div><div><b>PD:</b> {pdv:.3f}</div><div><b>Threshold:</b> {threshold:.2f}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Recomendação operacional**")
        st.write(suggestion)
        st.metric("Distância para o threshold", f"{abs(threshold - pdv):.3f}", delta="abaixo do limite" if pdv < threshold else "acima do limite")
        if TARGET_COL in row:
            st.caption(f"TARGET histórico: {int(row[TARGET_COL].iloc[0])}. Ele é exibido somente para comparação e não entra na decisão.")
    return scored


metrics = load_json("Model/metrics.json")
abt = load_csv(ABT_PATH)

with st.sidebar:
    st.markdown("## Política de crédito")
    st.caption("O threshold é o botão de apetite a risco da instituição.")
    threshold = st.slider("Threshold de decisão", 0.00, 1.00, float(RISK_THRESHOLD), 0.01, format="%.2f")
    st.markdown(
        f'<div class="threshold-note"><b>Regra ativa</b><br><br>PD &lt; <b>{threshold:.2f}</b> → CONCEDER<br>PD ≥ <b>{threshold:.2f}</b> → REVISAR / NÃO CONCEDER</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    if metrics:
        h = metrics.get("holdout", {})
        st.markdown("### Modelo em operação")
        st.write(f"**{metrics.get('best_model','-').replace('_',' ').title()}**")
        st.metric("AUC-ROC", fmt_num(h.get("auc_roc")))
        st.metric("KS", fmt_num(h.get("ks")))
    if st.button("Atualizar dados da tela", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown(
    '<div class="hero"><h1>💳 Crédito para pequenos empresários</h1><p>Simulação de concessão, diagnóstico do modelo, matriz de confusão e explicabilidade.</p></div>',
    unsafe_allow_html=True,
)
h = metrics.get("holdout", {}) if metrics else {}
c1, c2, c3, c4 = st.columns(4)
c1.metric("Modelo vencedor", metrics.get("best_model", "-").replace("_", " ").title() if metrics else "-")
c2.metric("AUC-ROC", fmt_num(h.get("auc_roc")))
c3.metric("KS", fmt_num(h.get("ks")))
c4.metric("Recall default", fmt_pct(h.get("recall_default")))

tabs = st.tabs(["🎯 Decisão de crédito", "🧪 Treinamento e resultados", "🧮 Matriz de confusão", "🔎 Explicabilidade", "🪣 Data lake"])
with tabs[0]:
    if abt is None or not storage.exists("Model/model.pkl"):
        st.warning("Rode a DAG até o final para gerar ABT e modelo.")
    else:
        mode = st.radio("Modo da simulação", ["Cliente histórico aleatório", "Nova solicitação simplificada"], horizontal=True)
        if mode == "Cliente histórico aleatório":
            a, b = st.columns([1.2, 4])
            with a:
                if st.button("🎲 Gerar cliente aleatório", type="primary", use_container_width=True):
                    st.session_state["row_idx"] = secrets.randbelow(len(abt))
            if "row_idx" not in st.session_state:
                st.info("Clique em **Gerar cliente aleatório**.")
            else:
                row = abt.iloc[[min(st.session_state.row_idx, len(abt) - 1)]].copy()
                show_decision(row, threshold, "ABT histórica")
                st.markdown("### Variáveis do cliente")
                specs = [
                    ("Renda total", "AMT_INCOME_TOTAL", "money"),
                    ("Crédito solicitado", "AMT_CREDIT", "money"),
                    ("Anuidade", "AMT_ANNUITY", "money"),
                    ("Idade", "AGE_YEARS", "num"),
                    ("Atraso médio", "INSTAL_DELAY_MEAN", "num"),
                    ("Maior atraso", "INSTAL_DPD_MAX", "num"),
                    ("Parcelas atrasadas", "INSTAL_LATE_PAYMENT_RATIO", "pct"),
                    ("Dívida / crédito", "BUREAU_DEBT_CREDIT_RATIO", "num"),
                ]
                vis = [x for x in specs if x[1] in row]
                for start in range(0, len(vis), 4):
                    cols = st.columns(4)
                    for col, (lab, key, kind) in zip(cols, vis[start:start + 4]):
                        with col:
                            v = row[key].iloc[0]
                            card(lab, money(v) if kind == "money" else (fmt_pct(v) if kind == "pct" else fmt_num(v, 2)), key)
                with st.expander("Registro completo"):
                    st.dataframe(row.T, use_container_width=True, height=500)
        else:
            st.caption("Campos não informados são imputados pelo pipeline. Histórico BUREAU/PREV/INSTAL é assumido como zero para uma solicitação sem histórico disponível.")
            with st.form("new_request"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    income = st.number_input("Renda anual", 0.0, 10_000_000.0, 180_000.0, 1_000.0)
                    credit = st.number_input("Crédito solicitado", 0.0, 10_000_000.0, 600_000.0, 1_000.0)
                    annuity = st.number_input("Anuidade/parcela", 0.0, 2_000_000.0, 27_000.0, 500.0)
                with col2:
                    goods = st.number_input("Valor do bem", 0.0, 10_000_000.0, 540_000.0, 1_000.0)
                    age = st.number_input("Idade", 18, 100, 35)
                    years = st.number_input("Anos de atividade", 0.0, 70.0, 5.0, .5)
                with col3:
                    ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, .51, .01)
                    ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, .57, .01)
                    ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, .54, .01)
                gender = st.selectbox("Gênero", ["F", "M"])
                education = st.selectbox("Escolaridade", ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"])
                income_type = st.selectbox("Tipo de renda", ["Commercial associate", "Businessman", "Working"])
                submitted = st.form_submit_button("Calcular score", type="primary", use_container_width=True)
            if submitted:
                row = pd.DataFrame([
                    {
                        "SK_ID_CURR": 999999,
                        "AMT_INCOME_TOTAL": income,
                        "AMT_CREDIT": credit,
                        "AMT_ANNUITY": annuity,
                        "AMT_GOODS_PRICE": goods,
                        "DAYS_BIRTH": -age * 365,
                        "DAYS_EMPLOYED": -years * 365,
                        "EXT_SOURCE_1": ext1,
                        "EXT_SOURCE_2": ext2,
                        "EXT_SOURCE_3": ext3,
                        "CODE_GENDER": gender,
                        "NAME_EDUCATION_TYPE": education,
                        "NAME_INCOME_TYPE": income_type,
                        "ORGANIZATION_TYPE": "Self-employed",
                        "SMALL_BUSINESS_PROXY": 1,
                    }
                ])
                show_decision(row, threshold, "nova solicitação")

with tabs[1]:
    st.subheader("Seleção e avaliação")
    comp = load_csv("reports/model_comparison.csv")
    if comp is not None:
        st.dataframe(comp, use_container_width=True)
    if metrics:
        sel = metrics.get("selection", {})
        hold = metrics.get("holdout", {})
        a, b, c, d = st.columns(4)
        a.metric("CV folds", sel.get("cv_folds", "-"))
        b.metric("Amostra da busca", fmt_pct(sel.get("search_sample_frac")))
        c.metric("CV-AUC ajustada", fmt_num(sel.get("best_grid_cv_auc")))
        d.metric("Holdout", f"{hold.get('rows',0):,}".replace(",", "."))
        st.markdown("#### Hiperparâmetros vencedores")
        st.json(sel.get("best_params", {}))
        roc = load_csv("reports/roc_curve_best_model.csv")
        if roc is not None:
            fig = px.line(roc, x="fpr", y="tpr", title="Curva ROC — holdout intocado")
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#64748b"))
            st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    hold = load_csv("reports/holdout_predictions.csv")
    if hold is None:
        st.info("Execute o treinamento.")
    else:
        y = hold[TARGET_COL].astype(int)
        p = hold.PD_DEFAULT.astype(float)
        pred = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        st.caption("Classe positiva = inadimplente. Falso negativo é o erro mais caro: inadimplente aprovado.")
        a, b, c, d = st.columns(4)
        a.metric("VN — bom aprovado", f"{tn:,}".replace(",", "."))
        b.metric("FP — bom revisado", f"{fp:,}".replace(",", "."))
        c.metric("FN — inadimplente aprovado", f"{fn:,}".replace(",", "."))
        d.metric("VP — inadimplente capturado", f"{tp:,}".replace(",", "."))
        matrix = pd.DataFrame([[tn, fp], [fn, tp]], index=["Real: bom", "Real: inadimplente"], columns=["Previsto: aprovar", "Previsto: revisar/negar"])
        fig = px.imshow(matrix, text_auto=True, aspect="auto", title=f"Matriz de confusão — threshold {threshold:.2f}")
        st.plotly_chart(fig, use_container_width=True)
        x, yc, z = st.columns(3)
        x.metric("Taxa de aprovação", fmt_pct((pred == 0).mean()))
        yc.metric("Recall inadimplente", fmt_pct(recall_score(y, pred, zero_division=0)))
        z.metric("Precision inadimplente", fmt_pct(precision_score(y, pred, zero_division=0)))
        ta = load_csv("reports/threshold_analysis.csv")
        if ta is not None:
            st.markdown("#### Comparação de políticas predefinidas")
            st.dataframe(ta, use_container_width=True)

with tabs[3]:
    st.subheader("O que mais influencia o modelo")
    native = load_csv("reports/feature_importance.csv")
    perm = load_csv("reports/permutation_importance.csv")
    shap = load_csv("reports/shap_importance.csv")
    for title, df, col in [
        ("Importância nativa", native, "abs_value"),
        ("Permutation importance", perm, "importance_mean"),
        ("SHAP médio absoluto", shap, "mean_abs_shap"),
    ]:
        if df is not None and col in df:
            top = df.head(20).sort_values(col)
            fig = px.bar(top, x=col, y="feature", orientation="h", title=title)
            fig.update_layout(height=550)
            st.plotly_chart(fig, use_container_width=True)
    if native is None and perm is None and shap is None:
        st.info("Execute o treinamento para gerar as análises.")

with tabs[4]:
    st.subheader("MinIO — fonte de verdade do pipeline")
    st.markdown(f"**Bucket:** `{MINIO_BUCKET}`  \n**Console:** http://localhost:9001")
    st.code("Dados/raw_data.csv\nDados/clean_data.csv\nDados/abt.csv\nDados/_processing/...\nModel/model.pkl\nModel/metrics.json\nreports/...", language="text")
    keys = storage.list_keys("") if storage.STORAGE_BACKEND == "minio" else []
    if keys:
        st.dataframe(pd.DataFrame({"object_key": keys}), use_container_width=True, height=400)
