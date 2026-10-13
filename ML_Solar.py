import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Intelligence Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

*, html, body { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #07090f;
    color: #c8cdd8;
}
h1,h2,h3 { font-family: 'Syne', sans-serif; }

[data-testid="stSidebar"] {
    background: #0c0f1a !important;
    border-right: 1px solid #1c2236;
}
header[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 1.5rem !important; }

.stTabs [data-baseweb="tab-list"] {
    background: #0c0f1a;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #1c2236;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 7px;
    color: #556080;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 13px;
    padding: 6px 18px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #1c2236 !important;
    color: #f0b429 !important;
}

.kpi-card {
    background: linear-gradient(145deg, #0e1221 0%, #0c1019 100%);
    border: 1px solid #1c2236;
    border-radius: 14px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
}
.kpi-eyebrow { font-size: 10px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #3d4f6e; margin-bottom: 8px; }
.kpi-val { font-family: 'Syne', sans-serif; font-size: 30px; font-weight: 800; color: var(--accent); line-height: 1; }
.kpi-desc { font-size: 11px; color: #3d4f6e; margin-top: 6px; }

.badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 10px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
}
.badge-gold { background: #f0b42922; color: #f0b429; border: 1px solid #f0b42944; }
.badge-blue { background: #63b3ed22; color: #63b3ed; border: 1px solid #63b3ed44; }
.badge-green { background: #68d39122; color: #68d391; border: 1px solid #68d39144; }
.badge-purple { background: #b794f422; color: #b794f4; border: 1px solid #b794f444; }

.model-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 20px; border-radius: 12px; margin-bottom: 10px;
    background: #0c0f1a; border: 1px solid #1c2236;
}
.model-row.best { border-color: #f0b42966; background: #f0b42908; }
.model-name { font-family: 'Syne', sans-serif; font-size: 15px; font-weight: 700; color: #c8cdd8; }
.model-stat { font-family: 'DM Mono', monospace; font-size: 14px; color: #f0b429; }
.model-stat-label { font-size: 10px; color: #3d4f6e; text-align: center; margin-top: 2px; }

.predict-panel {
    background: linear-gradient(145deg, #0e1221, #0c1019);
    border: 1px solid #1c2236; border-radius: 18px; padding: 28px 32px;
}
.output-orb {
    background: radial-gradient(circle at 50% 40%, #f0b42918 0%, transparent 70%);
    border: 1px solid #f0b42933; border-radius: 18px;
    padding: 32px 24px; text-align: center;
}
.orb-val { font-family: 'Syne', sans-serif; font-size: 52px; font-weight: 800; color: #f0b429; line-height: 1; }
.orb-unit { font-family: 'DM Mono', monospace; font-size: 14px; color: #556080; margin-top: 6px; letter-spacing: 0.08em; }
.orb-label { font-size: 11px; color: #3d4f6e; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px; }

.insight-box {
    background: #0e1221; border-left: 3px solid #f0b429;
    border-radius: 0 10px 10px 0; padding: 14px 18px; margin-bottom: 10px;
}
.insight-title { font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 700; color: #c8cdd8; margin-bottom: 4px; }
.insight-body { font-size: 12px; color: #556080; line-height: 1.6; }

label, .stSelectbox label, .stSlider label, .stNumberInput label {
    color: #556080 !important; font-size: 12px !important;
    font-weight: 500 !important; letter-spacing: 0.05em !important;
}
.stButton > button {
    background: linear-gradient(135deg, #f0b429, #e6930a);
    color: #07090f; font-family: 'Syne', sans-serif; font-weight: 700;
    border: none; border-radius: 10px; padding: 10px 24px;
    font-size: 14px; letter-spacing: 0.02em;
}
.stButton > button:hover { opacity: 0.88; }
div[data-testid="stExpander"] {
    background: #0c0f1a; border: 1px solid #1c2236 !important; border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 24px 0;'>
        <div style='font-family:Syne,sans-serif; font-size:20px; font-weight:800; color:#f0b429;'>☀ SOLAR</div>
        <div style='font-family:Syne,sans-serif; font-size:11px; font-weight:600; color:#3d4f6e; letter-spacing:0.14em;'>INTELLIGENCE DASHBOARD</div>
    </div>
    """, unsafe_allow_html=True)

    # ── File Upload ──────────────────────────
    st.markdown("**📂 UPLOAD DATASET**")
    uploaded_file = st.file_uploader(
        "Upload Solar CSV file", type=["csv"],
        help="CSV mein DC_POWER column hona zaroori hai. DATE_TIME optional hai."
    )
    if uploaded_file:
        st.markdown(f"""
        <div style='background:#68d39114; border:1px solid #68d39144; border-radius:8px;
                    padding:8px 12px; font-size:11px; color:#68d391; margin-top:6px;'>
            ✅ {uploaded_file.name}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#f0b42914; border:1px solid #f0b42944; border-radius:8px;
                    padding:8px 12px; font-size:11px; color:#f0b429; margin-top:6px;'>
            ⚠️ Pehle CSV file upload karo
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Model Settings ───────────────────────
    st.markdown("**⚙️ MODEL SETTINGS**")
    poly_degree  = st.slider("Polynomial Degree", 1, 4, 2,
                             help="Linear Reg aur Poly Reg dono ke liye")
    test_size    = st.slider("Test Split %", 10, 40, 30) / 100
    cv_folds     = st.slider("CV Folds", 3, 10, 5)
    dt_max_depth = st.slider("Decision Tree Max Depth", 2, 20, 5,
                             help="Zyada depth = overfitting ka risk")

    st.markdown("---")
    st.markdown("**🔽 DATA FILTERS**")
    remove_outliers = st.checkbox("Remove Outliers (IQR)", False)

    st.markdown("---")
    st.caption("Solar Power Predictor v3.0 · Linear · Polynomial · Decision Tree")


# ─────────────────────────────────────────────
# SPLASH SCREEN — agar file nahi upload ki
# ─────────────────────────────────────────────
if not uploaded_file:
    st.markdown("""
    <div style='display:flex; flex-direction:column; align-items:center; justify-content:center;
                height:65vh; text-align:center;'>
        <div style='font-size:64px; margin-bottom:20px;'>☀️</div>
        <div style='font-family:Syne,sans-serif; font-size:28px; font-weight:800;
                    color:#f0b429; margin-bottom:10px;'>
            Solar Intelligence Dashboard
        </div>
        <div style='color:#556080; font-size:14px; max-width:400px; line-height:1.8;'>
            Sidebar se apni <b style="color:#c8cdd8;">Solar CSV</b> file upload karo.<br>
            CSV mein <code style="color:#f0b429; background:#0c0f1a; padding:2px 6px;
            border-radius:4px;">DC_POWER</code> column hona zaroori hai.
        </div>
        <div style='margin-top:28px; background:#0c0f1a; border:1px dashed #2d3a54;
                    border-radius:14px; padding:20px 40px; color:#3d4f6e; font-size:12px;'>
            ← Left sidebar mein <b style="color:#c8cdd8;">📂 UPLOAD DATASET</b> section hai
        </div>
        <div style='margin-top:20px; display:flex; gap:12px; flex-wrap:wrap; justify-content:center;'>
            <span class='badge badge-gold'>Linear Regression</span>
            <span class='badge badge-blue'>Polynomial Regression</span>
            <span class='badge badge-purple'>Decision Tree Regression</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


SOLAR_SIGNATURE_COLUMNS = {
    "AC_POWER",
    "IRRADIATION",
    "DAILY_YIELD",
    "TOTAL_YIELD",
}


def validate_solar_dataset(df):
    """Ensure the uploaded CSV has the expected solar-power measurements."""
    columns = {column.strip().upper() for column in df.columns}
    if "DC_POWER" not in columns:
        raise ValueError(
            "Dataset is wrong. Solar CSV mein DC_POWER column hona zaroori hai."
        )

    signature_columns = columns.intersection(SOLAR_SIGNATURE_COLUMNS)
    if not signature_columns:
        raise ValueError(
            "Dataset is wrong. CSV solar-power dataset nahi lagti. "
            "AC_POWER, IRRADIATION, DAILY_YIELD ya TOTAL_YIELD mein se "
            "kam-se-kam ek column hona chahiye."
        )

    if df.empty:
        raise ValueError("Dataset is wrong. CSV file mein koi data row nahi hai.")

    numeric_feature_cols = [
        column for column in df.columns
        if column not in {"DC_POWER", "DATE_TIME", "SOURCE_KEY"}
        and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not numeric_feature_cols:
        raise ValueError(
            "Dataset is wrong. CSV mein kam-se-kam ek numeric solar feature "
            "hona chahiye."
        )

    required_numeric_columns = numeric_feature_cols + ["DC_POWER"]
    invalid_columns = [
        column for column in required_numeric_columns
        if not np.isfinite(pd.to_numeric(df[column], errors="coerce")).all()
    ]
    if invalid_columns:
        raise ValueError(
            "Dataset is wrong. Numeric columns mein blank, text ya invalid "
            f"value mili: {', '.join(invalid_columns)}."
        )


# ─────────────────────────────────────────────
# LOAD & TRAIN
# ─────────────────────────────────────────────
@st.cache_data
def load_and_train(file_bytes, file_name, poly_degree, test_size,
                   cv_folds, dt_max_depth, remove_outliers):

    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [column.strip() for column in df.columns]
    validate_solar_dataset(df)
    feature_cols = [
        column for column in df.columns
        if column not in {"DC_POWER", "DATE_TIME", "SOURCE_KEY"}
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    if remove_outliers:
        for col in feature_cols + ["DC_POWER"]:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    X = df[feature_cols].values
    y = df["DC_POWER"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=0)
    models_results = {}

    # ── 1. Linear Regression ─────────────────
    lr_pipe = Pipeline([("scl", StandardScaler()), ("lr", LinearRegression())])
    lr_pipe.fit(X_train, y_train)
    preds_lr = lr_pipe.predict(X_test)
    cv_lr = cross_val_score(lr_pipe, X, y, cv=kf, scoring="r2")
    models_results["Linear Regression"] = {
        "r2":    r2_score(y_test, preds_lr),
        "mae":   mean_absolute_error(y_test, preds_lr),
        "rmse":  np.sqrt(mean_squared_error(y_test, preds_lr)),
        "cv":    cv_lr.mean(),
        "cv_std":cv_lr.std(),
        "preds": preds_lr,
        "color": "#63b3ed",
    }

    # ── 2. Polynomial Regression ─────────────
    poly_obj = PolynomialFeatures(degree=poly_degree, include_bias=True)
    scaler   = StandardScaler()
    poly_lr  = LinearRegression()
    X_train_poly = scaler.fit_transform(poly_obj.fit_transform(X_train))
    X_test_poly  = scaler.transform(poly_obj.transform(X_test))
    poly_lr.fit(X_train_poly, y_train)
    preds_poly = poly_lr.predict(X_test_poly)
    cv_poly = cross_val_score(
        Pipeline([("poly", PolynomialFeatures(degree=poly_degree)),
                  ("scl",  StandardScaler()), ("lr", LinearRegression())]),
        X, y, cv=kf, scoring="r2"
    )
    models_results["Polynomial Regression"] = {
        "r2":    r2_score(y_test, preds_poly),
        "mae":   mean_absolute_error(y_test, preds_poly),
        "rmse":  np.sqrt(mean_squared_error(y_test, preds_poly)),
        "cv":    cv_poly.mean(),
        "cv_std":cv_poly.std(),
        "preds": preds_poly,
        "color": "#f0b429",
    }

    # ── 3. Decision Tree Regression ──────────
    dt = DecisionTreeRegressor(max_depth=dt_max_depth, random_state=42)
    dt.fit(X_train, y_train)
    preds_dt = dt.predict(X_test)
    cv_dt = cross_val_score(dt, X, y, cv=kf, scoring="r2")
    models_results["Decision Tree"] = {
        "r2":         r2_score(y_test, preds_dt),
        "mae":        mean_absolute_error(y_test, preds_dt),
        "rmse":       np.sqrt(mean_squared_error(y_test, preds_dt)),
        "cv":         cv_dt.mean(),
        "cv_std":     cv_dt.std(),
        "preds":      preds_dt,
        "color":      "#b794f4",
        "importances":dt.feature_importances_,
        "dt_obj":     dt,
    }

    best_name = max(models_results, key=lambda k: models_results[k]["r2"])

    return (df, feature_cols, X_train, X_test, y_train, y_test,
            poly_obj, scaler, poly_lr, lr_pipe, models_results, best_name)


try:
    with st.spinner("☀️ Models train ho rahe hain..."):
        (df, feature_cols, X_train, X_test, y_train, y_test,
         poly_obj, scaler, poly_lr, lr_pipe, models_results, best_name) = load_and_train(
            uploaded_file.getvalue(), uploaded_file.name,
            poly_degree, test_size, cv_folds, dt_max_depth, remove_outliers
        )
except ValueError as error:
    st.error(str(error))
    st.info(
        "Solar plant CSV upload karein. Expected columns mein DC_POWER ke "
        "saath AC_POWER, IRRADIATION, DAILY_YIELD ya TOTAL_YIELD hona chahiye."
    )
    st.stop()

best = models_results[best_name]


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
c_hdr, c_badge = st.columns([3, 1])
with c_hdr:
    st.markdown(f"""
    <h1 style='font-size:34px; font-weight:800; color:#f0b429; margin-bottom:2px;'>
        ☀ Solar Intelligence Dashboard
    </h1>
    <p style='color:#3d4f6e; font-size:13px; margin:0;'>
        DC Power Prediction · {len(df):,} records · {len(feature_cols)} features ·
        Best: <span style='color:#f0b429; font-weight:600;'>{best_name}</span>
    </p>
    """, unsafe_allow_html=True)
with c_badge:
    st.markdown(f"""
    <div style='text-align:right; padding-top:14px;'>
        <span class='badge badge-gold'>R² {best["r2"]:.4f}</span>&nbsp;
        <span class='badge badge-blue'>RMSE {best["rmse"]:.1f} W</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1c2236; margin:12px 0 20px 0;'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# KPI ROW  (best model ke metrics)
# ─────────────────────────────────────────────
kpi_data = [
    ("#f0b429", "R² Score",     f"{best['r2']:.4f}",    "Variance explained"),
    ("#63b3ed", "MAE",          f"{best['mae']:.1f} W",  "Mean absolute error"),
    ("#68d391", "RMSE",         f"{best['rmse']:.1f} W", "Root mean squared error"),
    ("#b794f4", "CV R² Mean",   f"{best['cv']:.4f}",     f"±{best['cv_std']:.4f}  ({cv_folds}-fold)"),
]
cols = st.columns(4)
for col, (accent, label, val, desc) in zip(cols, kpi_data):
    with col:
        st.markdown(f"""
        <div class='kpi-card' style='--accent:{accent};'>
            <div class='kpi-eyebrow'>{label}</div>
            <div class='kpi-val'>{val}</div>
            <div class='kpi-desc'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CHART THEME
# ─────────────────────────────────────────────
THEME = dict(
    paper_bgcolor="#07090f", plot_bgcolor="#0c0f1a",
    font=dict(color="#556080", family="DM Sans"),
    title_font=dict(family="Syne", color="#c8cdd8", size=15),
    margin=dict(l=24, r=24, t=48, b=24),
)
GRID = dict(gridcolor="#1c2236", zerolinecolor="#1c2236")
MODEL_COLORS = {
    "Linear Regression":      "#63b3ed",
    "Polynomial Regression":  "#f0b429",
    "Decision Tree":          "#b794f4",
}


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Predictions", "⚔️ Model Comparison",
    "🌳 Decision Tree", "📊 Data Analysis", "🔮 Live Predictor"
])


# ══════════════════════════════
# TAB 1 — PREDICTIONS
# ══════════════════════════════
with tab1:
    model_sel = st.selectbox(
        "Model select karo",
        list(models_results.keys()),
        index=list(models_results.keys()).index(best_name),
        key="tab1_model"
    )
    sel_preds = models_results[model_sel]["preds"]
    sel_color = MODEL_COLORS[model_sel]
    residuals_all = np.array(y_test) - sel_preds

    c1, c2 = st.columns([3, 2])
    with c1:
        n   = min(600, len(y_test))
        idx = np.random.choice(len(y_test), n, replace=False)
        act = np.array(y_test)[idx]
        pre = sel_preds[idx]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=act, y=pre, mode="markers",
            marker=dict(color=act, colorscale="Plasma", opacity=0.6, size=5,
                        colorbar=dict(title="Actual W", thickness=10, len=0.7)),
            name="Predictions",
            hovertemplate="Actual: %{x:.1f} W<br>Predicted: %{y:.1f} W<extra></extra>"
        ))
        mn, mx = float(min(act.min(), pre.min())), float(max(act.max(), pre.max()))
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            line=dict(color=sel_color, dash="dot", width=1.5), name="Perfect fit"
        ))
        fig.update_layout(**THEME, title=f"Actual vs Predicted · {model_sel}",
                          xaxis=dict(title="Actual (W)", **GRID),
                          yaxis=dict(title="Predicted (W)", **GRID), height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig_err = go.Figure()
        fig_err.add_trace(go.Histogram(
            x=residuals_all, nbinsx=55,
            marker_color=sel_color, opacity=0.75
        ))
        fig_err.add_vline(x=0, line_color="#fc4f4f", line_dash="dash", line_width=1.5)
        fig_err.update_layout(**THEME, title="Residual Distribution",
                               xaxis=dict(title="Error (W)", **GRID),
                               yaxis=dict(title="Frequency", **GRID), height=200)
        st.plotly_chart(fig_err, use_container_width=True)

        fig_rv = go.Figure()
        fig_rv.add_trace(go.Scatter(
            x=sel_preds[idx], y=residuals_all[idx], mode="markers",
            marker=dict(color=sel_color, opacity=0.4, size=4)
        ))
        fig_rv.add_hline(y=0, line_color="#f0b429", line_dash="dash", line_width=1.2)
        fig_rv.update_layout(**THEME, title="Residuals vs Predicted",
                              xaxis=dict(title="Predicted (W)", **GRID),
                              yaxis=dict(title="Residual (W)", **GRID), height=180)
        st.plotly_chart(fig_rv, use_container_width=True)

    with st.expander("📋 Error Percentile Analysis"):
        pct_df = pd.DataFrame({
            "Percentile": ["P10", "P25", "Median", "P75", "P90", "Max"],
            "Abs Error (W)": [np.percentile(np.abs(residuals_all), p)
                              for p in [10, 25, 50, 75, 90, 100]]
        }).round(2)
        st.dataframe(pct_df, use_container_width=True, hide_index=True)


# ══════════════════════════════
# TAB 2 — MODEL COMPARISON
# ══════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)

    sorted_models = sorted(models_results.items(), key=lambda x: x[1]["r2"], reverse=True)
    badge_classes = ["badge-gold", "badge-blue", "badge-purple"]

    for i, (name, res) in enumerate(sorted_models):
        is_best = (name == best_name)
        b_cls = "badge-gold" if is_best else badge_classes[min(i, 2)]
        b_txt = "BEST ✨" if is_best else f"#{i+1}"
        st.markdown(f"""
        <div class='model-row {"best" if is_best else ""}'>
            <div style='display:flex; align-items:center; gap:12px;'>
                <span class='badge {b_cls}'>{b_txt}</span>
                <span class='model-name'>{name}</span>
            </div>
            <div style='display:flex; gap:36px;'>
                <div style='text-align:center;'>
                    <div class='model-stat' style='color:{res["color"]};'>{res["r2"]:.4f}</div>
                    <div class='model-stat-label'>R²</div>
                </div>
                <div style='text-align:center;'>
                    <div class='model-stat' style='color:#63b3ed;'>{res["mae"]:.1f}</div>
                    <div class='model-stat-label'>MAE (W)</div>
                </div>
                <div style='text-align:center;'>
                    <div class='model-stat' style='color:#68d391;'>{res["rmse"]:.1f}</div>
                    <div class='model-stat-label'>RMSE (W)</div>
                </div>
                <div style='text-align:center;'>
                    <div class='model-stat' style='color:#b794f4;'>{res["cv"]:.4f} ±{res["cv_std"]:.4f}</div>
                    <div class='model-stat-label'>CV R² ({cv_folds}-fold)</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c_bar, c_cv = st.columns(2)
    with c_bar:
        max_mae  = max(r["mae"]  for r in models_results.values())
        max_rmse = max(r["rmse"] for r in models_results.values())
        fig_bar = go.Figure()
        for name, res in models_results.items():
            fig_bar.add_trace(go.Bar(
                name=name,
                x=["R²", "1−Norm.MAE", "1−Norm.RMSE"],
                y=[res["r2"], 1 - res["mae"]/max_mae, 1 - res["rmse"]/max_rmse],
                marker_color=res["color"], opacity=0.85
            ))
        fig_bar.update_layout(**THEME, title="Normalised Metric Comparison",
                               barmode="group", height=340,
                               yaxis=dict(title="Score (higher = better)", **GRID),
                               xaxis=dict(**GRID),
                               legend=dict(bgcolor="#0c0f1a", bordercolor="#1c2236"))
        st.plotly_chart(fig_bar, use_container_width=True)

    with c_cv:
        fig_cv = go.Figure()
        for name, res in models_results.items():
            fig_cv.add_trace(go.Bar(
                name=name, x=[name], y=[res["cv"]],
                error_y=dict(type="data", array=[res["cv_std"]], visible=True, color="#556080"),
                marker_color=res["color"], opacity=0.85
            ))
        fig_cv.update_layout(**THEME, title=f"Cross-Validation R² ({cv_folds}-fold)",
                              showlegend=False, height=340,
                              yaxis=dict(title="CV R²", **GRID), xaxis=dict(**GRID))
        st.plotly_chart(fig_cv, use_container_width=True)

    # Overlay predictions
    with st.expander("🔍 Teen Models ki Predictions Ek Saath Dekho"):
        n2  = min(300, len(y_test))
        idx2 = np.sort(np.random.choice(len(y_test), n2, replace=False))
        act2 = np.array(y_test)[idx2]

        fig_ov = go.Figure()
        fig_ov.add_trace(go.Scatter(
            x=list(range(n2)), y=act2, mode="lines",
            line=dict(color="#ffffff", width=1.5, dash="dot"), name="Actual"
        ))
        for name, res in models_results.items():
            fig_ov.add_trace(go.Scatter(
                x=list(range(n2)), y=res["preds"][idx2], mode="lines",
                line=dict(color=res["color"], width=1.5), name=name, opacity=0.85
            ))
        fig_ov.update_layout(**THEME, title="Actual vs All 3 Models (300 samples)",
                              xaxis=dict(title="Sample Index", **GRID),
                              yaxis=dict(title="DC Power (W)", **GRID),
                              height=380,
                              legend=dict(bgcolor="#0c0f1a", bordercolor="#1c2236"))
        st.plotly_chart(fig_ov, use_container_width=True)


# ══════════════════════════════
# TAB 3 — DECISION TREE
# ══════════════════════════════
with tab3:
    dt_obj = models_results["Decision Tree"]["dt_obj"]
    dt_imps = models_results["Decision Tree"]["importances"]

    c_fi, c_corr = st.columns(2)

    with c_fi:
        fi_df = pd.DataFrame({
            "Feature":    feature_cols,
            "Importance": dt_imps
        }).sort_values("Importance", ascending=True)

        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"],
            orientation="h",
            marker=dict(
                color=fi_df["Importance"],
                colorscale=[[0, "#1c2236"], [0.5, "#b794f4"], [1, "#f0b429"]],
                showscale=False
            )
        ))
        fig_fi.update_layout(**THEME, title="Decision Tree · Feature Importance",
                              xaxis=dict(title="Importance Score", **GRID),
                              yaxis=dict(**GRID), height=360)
        st.plotly_chart(fig_fi, use_container_width=True)

    with c_corr:
        corr_df = pd.DataFrame(X_train, columns=feature_cols)
        corr_df["DC_POWER"] = y_train
        corr_m = corr_df.corr()
        fig_hm = go.Figure(go.Heatmap(
            z=corr_m.values,
            x=corr_m.columns, y=corr_m.index,
            colorscale=[[0, "#1c2236"], [0.5, "#4a5f8a"], [1, "#f0b429"]],
            zmid=0,
            text=corr_m.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=10, color="#c8cdd8"),
            hovertemplate="%{x} × %{y}: %{z:.3f}<extra></extra>"
        ))
        fig_hm.update_layout(**THEME, title="Feature Correlation Heatmap",
                             height=360,
                             xaxis=dict(tickangle=-35, **GRID),
                             yaxis=dict(**GRID))
        st.plotly_chart(fig_hm, use_container_width=True)

    # Depth vs R² chart
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📉 Decision Tree Depth vs R² (Overfitting Analysis)"):
        depths = list(range(1, 21))
        r2_train_list, r2_test_list = [], []
        X_tr, X_te, y_tr, y_te = train_test_split(
            df[feature_cols].values,
            df["DC_POWER"].values, test_size=test_size, random_state=42
        )
        for d in depths:
            t = DecisionTreeRegressor(max_depth=d, random_state=42)
            t.fit(X_tr, y_tr)
            r2_train_list.append(r2_score(y_tr, t.predict(X_tr)))
            r2_test_list.append(r2_score(y_te, t.predict(X_te)))

        fig_depth = go.Figure()
        fig_depth.add_trace(go.Scatter(
            x=depths, y=r2_train_list, mode="lines+markers",
            name="Train R²", line=dict(color="#68d391", width=2),
            marker=dict(size=6)
        ))
        fig_depth.add_trace(go.Scatter(
            x=depths, y=r2_test_list, mode="lines+markers",
            name="Test R²", line=dict(color="#b794f4", width=2),
            marker=dict(size=6)
        ))
        fig_depth.add_vline(x=dt_max_depth, line_color="#f0b429",
                            line_dash="dash", line_width=1.5,
                            annotation_text=f"Current depth={dt_max_depth}",
                            annotation_font_color="#f0b429")
        fig_depth.update_layout(**THEME, title="Depth vs R² (Overfit/Underfit Check)",
                                xaxis=dict(title="Max Depth", **GRID),
                                yaxis=dict(title="R² Score", **GRID),
                                legend=dict(bgcolor="#0c0f1a", bordercolor="#1c2236"),
                                height=340)
        st.plotly_chart(fig_depth, use_container_width=True)

    # Tree text rules
    with st.expander("📋 Decision Tree Rules (Text)"):
        tree_rules = export_text(dt_obj, feature_names=feature_cols, max_depth=4)
        st.code(tree_rules, language="text")


# ══════════════════════════════
# TAB 4 — DATA ANALYSIS
# ══════════════════════════════
with tab4:
    c_sel, _ = st.columns([2, 2])
    with c_sel:
        feat_sel = st.selectbox("Feature select karo", feature_cols)

    c_hist, c_box = st.columns(2)
    with c_hist:
        fig_h = go.Figure(go.Histogram(
            x=df[feat_sel], nbinsx=50, marker_color="#f0b429", opacity=0.75
        ))
        fig_h.update_layout(**THEME, title=f"Distribution · {feat_sel}",
                            xaxis=dict(**GRID), yaxis=dict(**GRID), height=300)
        st.plotly_chart(fig_h, use_container_width=True)

    with c_box:
        fig_bx = go.Figure(go.Box(
            y=df[feat_sel], marker_color="#f0b429", line_color="#f0b429",
            fillcolor="#f0b42922", boxpoints="outliers", jitter=0.3,
            marker=dict(opacity=0.5, size=3)
        ))
        fig_bx.update_layout(**THEME, title=f"Box Plot · {feat_sel}",
                             yaxis=dict(**GRID), height=300)
        st.plotly_chart(fig_bx, use_container_width=True)

    fig_sc = px.scatter(
        df, x=feat_sel, y="DC_POWER",
        color="DC_POWER", color_continuous_scale="Plasma",
        opacity=0.5, trendline="lowess",
        title=f"{feat_sel} vs DC Power (LOWESS trend)"
    )
    fig_sc.update_layout(**THEME, height=350,
                          xaxis=dict(**GRID), yaxis=dict(**GRID))
    st.plotly_chart(fig_sc, use_container_width=True)

    with st.expander("📋 Dataset Statistics"):
        st.dataframe(df[feature_cols + ["DC_POWER"]].describe().T.round(3),
                     use_container_width=True)

    with st.expander("🗂 Raw Data (first 200 rows)"):
        st.dataframe(df.head(200), use_container_width=True, height=260)


# ══════════════════════════════
# TAB 5 — LIVE PREDICTOR
# ══════════════════════════════
with tab5:
    st.markdown("<br>", unsafe_allow_html=True)
    c_form, c_out = st.columns([3, 2])

    with c_form:
        st.markdown('<div class="predict-panel">', unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family:Syne,sans-serif; font-size:16px; font-weight:700;
                    color:#c8cdd8; margin-bottom:6px;'>⚡ Sensor Values Adjust Karo</div>
        """, unsafe_allow_html=True)

        pred_model_sel = st.selectbox(
            "Prediction model",
            list(models_results.keys()),
            index=list(models_results.keys()).index(best_name),
            key="pred_model"
        )

        user_inputs = {}
        for feat in feature_cols:
            fmin  = float(df[feat].min())
            fmax  = float(df[feat].max())
            fmean = float(df[feat].mean())
            if fmin == fmax:
                st.caption(f"{feat}: {fmin:,.3f} (constant value)")
                user_inputs[feat] = fmin
            else:
                step = round((fmax - fmin) / 300, 4) or 0.0001
                user_inputs[feat] = st.slider(
                    feat, min_value=fmin, max_value=fmax,
                    value=fmean, step=step, format="%.3f",
                    key=f"slider_{feat}"
                )

        st.button("⚡ Predict", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_out:
        input_arr = np.array([[user_inputs[f] for f in feature_cols]])

        if pred_model_sel == "Linear Regression":
            prediction = max(0.0, float(lr_pipe.predict(input_arr)[0]))
        elif pred_model_sel == "Polynomial Regression":
            inp_poly   = scaler.transform(poly_obj.transform(input_arr))
            prediction = max(0.0, float(poly_lr.predict(inp_poly)[0]))
        else:  # Decision Tree
            prediction = max(0.0, float(models_results["Decision Tree"]["dt_obj"].predict(input_arr)[0]))

        pct    = (prediction / float(df["DC_POWER"].max())) * 100
        m_rmse = models_results[pred_model_sel]["rmse"]
        m_r2   = models_results[pred_model_sel]["r2"]
        m_col  = models_results[pred_model_sel]["color"]

        st.markdown(f"""
        <div class="output-orb" style='margin-bottom:16px; border-color:{m_col}33;
             background: radial-gradient(circle at 50% 40%, {m_col}18 0%, transparent 70%);'>
            <div class="orb-label">Predicted DC Power Output</div>
            <div class="orb-val" style='color:{m_col};'>{prediction:,.0f}</div>
            <div class="orb-unit">Watts</div>
            <div style='margin-top:10px;'>
                <span class='badge' style='background:{m_col}22; color:{m_col};
                    border:1px solid {m_col}44;'>{pred_model_sel}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number=dict(suffix="%", font=dict(size=28, color=m_col, family="Syne")),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#556080",
                          tickfont=dict(color="#556080", size=10)),
                bar=dict(color=m_col),
                bgcolor="#0c0f1a", bordercolor="#1c2236", borderwidth=1,
                steps=[
                    dict(range=[0, 33],  color="#1c2236"),
                    dict(range=[33, 66], color="#2d3a54"),
                    dict(range=[66, 100],color="#3d4f6e"),
                ],
                threshold=dict(line=dict(color="#fc8181", width=2), thickness=0.75, value=90)
            ),
            title=dict(text="% of Peak Capacity", font=dict(color="#3d4f6e", size=11)),
            domain=dict(x=[0, 1], y=[0, 1])
        ))
        fig_g.update_layout(paper_bgcolor="#07090f", font_color="#556080",
                            height=200, margin=dict(l=16, r=16, t=32, b=8))
        st.plotly_chart(fig_g, use_container_width=True)

        level = "High ☀️" if pct > 60 else ("Moderate 🌤️" if pct > 30 else "Low 🌑")
        st.markdown(f"""
        <div class="insight-box" style='border-left-color:{m_col};'>
            <div class="insight-title">Output Level · {level}</div>
            <div class="insight-body">
                {pct:.1f}% of peak ({df["DC_POWER"].max():.0f} W) ·
                Model R² = {m_r2:.4f}
            </div>
        </div>
        <div class="insight-box" style='border-left-color:#63b3ed;'>
            <div class="insight-title">Confidence Range (±1 RMSE)</div>
            <div class="insight-body">
                <b style='color:#63b3ed;'>
                {max(0, prediction - m_rmse):.0f} W — {prediction + m_rmse:.0f} W
                </b>
                &nbsp;(RMSE = {m_rmse:.1f} W)
            </div>
        </div>
        """, unsafe_allow_html=True)


st.markdown("""
<div style='text-align:center; padding:32px 0 8px 0; color:#1c2236;
            font-size:11px; font-family: DM Mono, monospace; letter-spacing:0.08em;'>
    SOLAR INTELLIGENCE DASHBOARD · LINEAR · POLYNOMIAL · DECISION TREE · STREAMLIT + PLOTLY
</div>
""", unsafe_allow_html=True)
