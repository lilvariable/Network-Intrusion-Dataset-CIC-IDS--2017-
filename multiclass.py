import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Class Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'dashboard'
VIS_DIR  = BASE_DIR / 'visuals'

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    class_dist   = pd.read_csv(DATA_DIR / 'multiclass_class_distribution.csv')
    model_comp   = pd.read_csv(DATA_DIR / 'multiclass_model_comparison.csv')
    rf_feat      = pd.read_csv(DATA_DIR / 'rf_multiclass_feature_importance.csv')
    dt_feat      = pd.read_csv(DATA_DIR / 'dt_multiclass_feature_importance.csv')
    lr_coef      = pd.read_csv(DATA_DIR / 'lr_coefficients.csv')
    per_class    = pd.read_csv(DATA_DIR / 'multiclass_per_class_metrics.csv')
    predictions  = pd.read_csv(DATA_DIR / 'multiclass_predictions.csv')
    return class_dist, model_comp, rf_feat, dt_feat, lr_coef, per_class, predictions

class_dist, model_comp, rf_feat, dt_feat, lr_coef, per_class, predictions = load_data()

# ── Header ────────────────────────────────────────────────────
st.title("🛡️ Multi-Class Network Intrusion Detection Dashboard")
st.markdown("**Cyber Threat Intelligence Project** — 12-Class Attack Detection using Machine Learning")
st.divider()

# ── Section 1: Dataset Overview ───────────────────────────────
st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Flows", "2,830,675")
col2.metric("Features Used", "47")
col3.metric("Attack Types", "11")
col4.metric("Training Samples", "2,264,540")

st.subheader("Class Distribution")
scale = st.radio("Y-axis scale", ["Linear", "Log"], horizontal=True)

fig_dist = px.bar(
    class_dist.sort_values('Count', ascending=False),
    x='Label', y='Count',
    color='Label',
    text='Count'
)
fig_dist.update_traces(textposition='outside', texttemplate='%{text:,}')
fig_dist.update_layout(
    showlegend=False,
    height=450,
    xaxis_tickangle=-35,
    yaxis_type='log' if scale == 'Log' else 'linear'
)
st.plotly_chart(fig_dist, use_container_width=True)
st.divider()

# ── Section 2: Feature Analysis ───────────────────────────────
st.header("🔍 Feature Analysis")

model_choice = st.selectbox(
    "Select model for feature importance",
    ["Random Forest", "Decision Tree"]
)

feat_df = rf_feat if model_choice == "Random Forest" else dt_feat

# Rename columns defensively
feat_df.columns = ['Feature', 'Importance']

fig_feat = px.bar(
    feat_df.head(15),
    x='Importance', y='Feature',
    orientation='h',
    color='Importance',
    color_continuous_scale='Blues',
    title=f"{model_choice} — Top 15 Feature Importances"
)
fig_feat.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
st.plotly_chart(fig_feat, use_container_width=True)

# Logistic Regression coefficients
st.subheader("Logistic Regression — Feature Coefficients by Class")
selected_class = st.selectbox("Select attack class", sorted(lr_coef['Class'].unique()))
coef_filtered = lr_coef[lr_coef['Class'] == selected_class].sort_values('Coefficient')
top_coef = pd.concat([coef_filtered.head(10), coef_filtered.tail(10)])

fig_coef = px.bar(
    top_coef,
    x='Coefficient', y='Feature',
    orientation='h',
    color='Coefficient',
    color_continuous_scale='RdBu',
    title=f"Top Positive & Negative Coefficients — {selected_class}"
)
fig_coef.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
st.plotly_chart(fig_coef, use_container_width=True)
st.divider()

# ── Section 3: Model Performance ─────────────────────────────
st.header("🤖 Model Performance")

# Summary metrics table
st.subheader("Overall Performance Metrics")
styled_df = model_comp.set_index('Model').round(4)

def highlight_all_max(df):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    for col in df.columns:
        max_val = df[col].max()
        for idx in df.index:
            if df.loc[idx, col] == max_val:
                styles.loc[idx, col] = 'background-color: #013220'
    return styles

st.dataframe(
    styled_df.style.format("{:.4f}").apply(highlight_all_max, axis=None),
    use_container_width=True
)

# Bar chart
model_melted = model_comp.melt(id_vars='Model', var_name='Metric', value_name='Score')
fig_models = px.bar(
    model_melted,
    x='Model', y='Score', color='Metric',
    barmode='group',
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig_models.update_layout(height=450)
st.plotly_chart(fig_models, use_container_width=True)

# Per-class RF metrics
st.subheader("Random Forest — Per-Class Performance")
per_class_display = per_class.rename(columns={'Unnamed: 0': 'Class'}).set_index('Class')
per_class_display = per_class_display[['precision', 'recall', 'f1-score', 'support']].round(4)

fig_perclass = px.bar(
    per_class.rename(columns={'Unnamed: 0': 'Class'}),
    x='Class', y='f1-score',
    color='f1-score',
    color_continuous_scale='RdYlGn',
    range_color=[0, 1],
    title='F1-Score per Attack Class — Random Forest'
)
fig_perclass.update_layout(xaxis_tickangle=-35, height=450)
st.plotly_chart(fig_perclass, use_container_width=True)

st.dataframe(
    per_class_display.style.format("{:.4f}", subset=['precision', 'recall', 'f1-score'])
    .format("{:.0f}", subset=['support']),
    use_container_width=True
)

# Confusion matrices
st.subheader("Confusion Matrices")
cm_choice = st.selectbox(
    "Select model",
    ["Random Forest", "Logistic Regression", "Decision Tree"]
)
cm_files = {
    "Random Forest": "rf_multiclass_confusion_matrix.png",
    "Logistic Regression": "lr_multiclass_confusion_matrix.png",
    "Decision Tree": "dt_multiclass_confusion_matrix.png"
}
cm_path = VIS_DIR / cm_files[cm_choice]
if cm_path.exists():
    st.image(str(cm_path), use_container_width=True)
else:
    st.warning(f"Confusion matrix image not found: {cm_path}")
st.divider()

# ── Section 4: Threat Explorer ────────────────────────────────
st.header("🚨 Threat Explorer")
st.markdown("Explore individual flow predictions from the Random Forest model.")

col1, col2 = st.columns(2)
with col1:
    label_filter = st.selectbox(
        "Filter by True Label",
        ["All"] + sorted(predictions['True_Label'].unique().tolist())
    )
with col2:
    correct_filter = st.selectbox(
        "Prediction outcome",
        ["All", "Correct only", "Misclassified only"]
    )

display_df = predictions.copy()
if label_filter != "All":
    display_df = display_df[display_df['True_Label'] == label_filter]
if correct_filter == "Correct only":
    display_df = display_df[display_df['Correct'] == 1]
elif correct_filter == "Misclassified only":
    display_df = display_df[display_df['Correct'] == 0]

st.dataframe(
    display_df[['True_Label', 'RF_Prediction', 'Correct']].head(500),
    use_container_width=True
)
st.caption(f"Showing {min(500, len(display_df))} of {len(display_df)} rows")