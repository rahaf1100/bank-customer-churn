from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# DESIGN SYSTEM
# =========================================================

APP_TITLE = "Bank Customer Analytics"
APP_SUBTITLE = "Customer Intelligence & Churn Prediction"

NAVY = "#1D3354"
BLUE = "#467599"
TEAL = "#9ED8DB"
RED = "#D64045"
BG = "#F4F7FB"
WHITE = "#FFFFFF"
TEXT = "#26364A"
MUTED = "#718096"
BORDER = "#E2E8F0"
GRID = "#EDF1F6"

FONT = "Arial, sans-serif"
CONTENT_WIDTH = "1480px"
PAGE_TOP_PADDING = "1.8rem"
CARD_RADIUS = "14px"
CARD_PADDING = "20px"

CHURN_THRESHOLD = 0.35

CHART_COLORS = {
    "Stayed": BLUE,
    "Churned": RED,
    "Active": TEAL,
    "Inactive": NAVY,
    "France": NAVY,
    "Germany": BLUE,
    "Spain": TEAL,
}

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "churn_model.joblib"
DATA_PATH = BASE_DIR / "Churn_Modelling.csv"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# GLOBAL STYLING
# =========================================================

st.markdown(
    f"""
    <style>
    :root {{
        --navy: {NAVY};
        --blue: {BLUE};
        --teal: {TEAL};
        --red: {RED};
        --bg: {BG};
        --white: {WHITE};
        --text: {TEXT};
        --muted: {MUTED};
        --border: {BORDER};
    }}

    html, body, [class*="css"] {{
        font-family: {FONT};
    }}

    .stApp {{
        background: var(--bg);
        color: var(--text);
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    .block-container {{
        max-width: {CONTENT_WIDTH};
        padding-top: {PAGE_TOP_PADDING};
        padding-bottom: 3rem;
    }}

    h1, h2, h3 {{
        color: var(--navy) !important;
        letter-spacing: -0.35px;
    }}

    h1 {{
        font-size: 2rem !important;
        font-weight: 750 !important;
    }}

    h2 {{
        font-size: 1.35rem !important;
        font-weight: 700 !important;
    }}

    h3 {{
        font-size: 1.05rem !important;
        font-weight: 650 !important;
    }}

    p, label {{
        color: var(--text);
    }}

    .page-subtitle {{
        color: var(--muted);
        font-size: 0.94rem;
        margin-top: -0.7rem;
        margin-bottom: 1.8rem;
        line-height: 1.6;
    }}

    .eyebrow {{
        color: var(--blue);
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }}

    .section-heading {{
        font-size: 1.1rem;
        color: var(--navy);
        font-weight: 750;
        margin: 1.3rem 0 0.8rem 0;
    }}

    /* Sidebar */

    [data-testid="stSidebar"] {{
        background: var(--navy);
        border-right: 1px solid rgba(255,255,255,0.08);
    }}

    [data-testid="stSidebar"] * {{
        color: #F7FAFC;
    }}

    [data-testid="stSidebar"] .block-container {{
        padding-top: 1.5rem;
    }}

    .brand-mark {{
        width: 43px;
        height: 43px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(158,216,219,0.14);
        border: 1px solid rgba(158,216,219,0.35);
        border-radius: 12px;
        color: {TEAL};
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }}

    .sidebar-brand {{
        color: #FFFFFF;
        font-size: 1.1rem;
        font-weight: 800;
        letter-spacing: 0.7px;
    }}

    .sidebar-caption {{
        color: #C5D2E0;
        font-size: 0.77rem;
        line-height: 1.5;
        margin-top: 0.35rem;
        margin-bottom: 1.5rem;
    }}

    [data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.15);
    }}

    /* Sidebar navigation buttons — fixed text contrast */

    [data-testid="stSidebar"] div.stButton > button {{
        background-color: transparent !important;
        color: #C5D2E0 !important;
        border: 1px solid transparent !important;
        border-radius: 9px;
        padding: 0.72rem 0.8rem;
        text-align: left;
        justify-content: flex-start;
        min-height: 46px;
        font-weight: 600;
        box-shadow: none !important;
        transition: background 0.15s ease;
    }}

    [data-testid="stSidebar"] div.stButton > button p,
    [data-testid="stSidebar"] div.stButton > button span {{
        color: #C5D2E0 !important;
    }}

    [data-testid="stSidebar"] div.stButton > button:hover {{
        background-color: rgba(255,255,255,0.12) !important;
        border-color: rgba(255,255,255,0.15) !important;
    }}

    [data-testid="stSidebar"] div.stButton > button:hover p,
    [data-testid="stSidebar"] div.stButton > button:hover span {{
        color: #FFFFFF !important;
    }}

    [data-testid="stSidebar"] div.stButton > button[kind="primary"] {{
        background-color: #FFFFFF !important;
        color: #1D3354 !important;
        border-color: #FFFFFF !important;
        font-weight: 750;
    }}

    [data-testid="stSidebar"] div.stButton > button[kind="primary"] p,
    [data-testid="stSidebar"] div.stButton > button[kind="primary"] span {{
        color: #1D3354 !important;
    }}

    [data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover {{
        background-color: #F4F7FB !important;
        color: #1D3354 !important;
        border-color: #F4F7FB !important;
    }}

    /* General buttons — readable white text */

    .stApp div.stButton > button,
    .stApp div.stFormSubmitButton > button {{
        background-color: #1D3354 !important;
        color: #FFFFFF !important;
        border: 1px solid #1D3354 !important;
        border-radius: 9px;
        min-height: 43px;
        font-weight: 650;
        transition: all 0.15s ease;
    }}

    .stApp div.stButton > button p,
    .stApp div.stButton > button span,
    .stApp div.stFormSubmitButton > button p,
    .stApp div.stFormSubmitButton > button span {{
        color: #FFFFFF !important;
    }}

    .stApp div.stButton > button:hover,
    .stApp div.stFormSubmitButton > button:hover {{
        background-color: #467599 !important;
        border-color: #467599 !important;
        color: #FFFFFF !important;
    }}

    .stApp div.stButton > button:hover p,
    .stApp div.stButton > button:hover span,
    .stApp div.stFormSubmitButton > button:hover p,
    .stApp div.stFormSubmitButton > button:hover span {{
        color: #FFFFFF !important;
    }}

    /* Preserve the white selected navigation button */

    [data-testid="stSidebar"] div.stButton > button[kind="primary"],
    [data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover {{
        background-color: #FFFFFF !important;
        color: #1D3354 !important;
        border-color: #FFFFFF !important;
    }}

    [data-testid="stSidebar"] div.stButton > button[kind="primary"] p,
    [data-testid="stSidebar"] div.stButton > button[kind="primary"] span {{
        color: #1D3354 !important;
    }}

    /* KPI cards */

    [data-testid="stMetric"] {{
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: {CARD_RADIUS};
        padding: 1.15rem 1.25rem;
        min-height: 125px;
        box-shadow: 0 2px 7px rgba(29,51,84,0.025);
    }}

    [data-testid="stMetricLabel"] {{
        color: var(--muted) !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }}

    [data-testid="stMetricValue"] {{
        color: var(--navy) !important;
        font-size: 1.8rem !important;
        font-weight: 750 !important;
    }}

    [data-testid="stMetricDelta"] {{
        font-size: 0.75rem;
    }}

    /* Charts and tables */

    [data-testid="stPlotlyChart"] {{
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: {CARD_RADIUS};
        padding: 0.45rem 0.45rem 0.1rem 0.45rem;
        box-shadow: 0 2px 7px rgba(29,51,84,0.025);
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }}

    /* Inputs */

    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {{
        border-color: var(--border);
        border-radius: 8px;
        background: var(--white);
    }}

    div.stButton > button,
    div.stFormSubmitButton > button {{
        background: var(--navy);
        color: #FFFFFF;
        border: 1px solid var(--navy);
        border-radius: 9px;
        min-height: 43px;
        font-weight: 650;
        transition: all 0.15s ease;
    }}

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {{
        background: var(--blue);
        border-color: var(--blue);
        color: #FFFFFF;
    }}

    [data-testid="stSidebar"] div.stButton > button[kind="primary"] {{
        background: #FFFFFF;
        color: #1D3354;
        border-color: #FFFFFF;
    }}

    .info-card {{
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: {CARD_RADIUS};
        padding: {CARD_PADDING};
        height: 100%;
    }}

    .info-label {{
        color: var(--blue);
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 1.1px;
        text-transform: uppercase;
    }}

    .info-value {{
        color: var(--navy);
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: 0.5rem;
        margin-bottom: 0.35rem;
    }}

    .info-description {{
        color: var(--muted);
        font-size: 0.84rem;
        line-height: 1.6;
    }}

    .risk-panel {{
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: {CARD_RADIUS};
        padding: 1.4rem;
    }}

    .footer-note {{
        color: var(--muted);
        font-size: 0.76rem;
        line-height: 1.6;
        padding-top: 1.2rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODEL AND DATA
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    model = load_model()
    df = load_data()
except Exception as error:
    st.error(f"Unable to load the model or dataset: {error}")
    st.stop()


required_columns = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "Exited",
]

missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()


# =========================================================
# REUSABLE COMPONENTS
# =========================================================

def page_header(title, subtitle):
    st.markdown(
        '<div class="eyebrow">CUSTOMER INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )
    st.title(title)
    st.markdown(
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True,
    )


def info_card(label, value, description):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">{label}</div>
            <div class="info-value">{value}</div>
            <div class="info-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_chart(fig, height=350, horizontal=False):
    fig.update_layout(
        template="plotly_white",
        font=dict(family=FONT, color=TEXT, size=12),
        title=dict(
            font=dict(size=16, color=NAVY),
            x=0.02,
            xanchor="left",
        ),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        colorway=[NAVY, BLUE, TEAL, RED],
        height=height,
        margin=dict(l=18, r=22, t=68, b=35),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=MUTED),
        ),
        hoverlabel=dict(
            bgcolor=WHITE,
            bordercolor=BORDER,
            font=dict(color=TEXT, family=FONT),
        ),
    )

    fig.update_xaxes(
        showgrid=not horizontal,
        gridcolor=GRID,
        zeroline=False,
        linecolor=BORDER,
        tickfont=dict(color=MUTED, size=11),
        title_font=dict(color=MUTED, size=11),
    )

    fig.update_yaxes(
        showgrid=horizontal,
        gridcolor=GRID,
        zeroline=False,
        linecolor=BORDER,
        tickfont=dict(color=MUTED, size=11),
        title_font=dict(color=MUTED, size=11),
    )

    return fig


def show_chart(fig, height=350, horizontal=False):
    st.plotly_chart(
        style_chart(fig, height=height, horizontal=horizontal),
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )


def format_percent(value):
    return f"{value:.1f}%"


# =========================================================
# SIDEBAR NAVIGATION — CLICKABLE PAGE BUTTONS
# =========================================================

PAGES = [
    "Dashboard",
    "Churn Prediction",
    "Model Insights",
    "Customer Analytics",
]


def set_page(selected_page):
    st.session_state["current_page"] = selected_page


if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Dashboard"


with st.sidebar:
    st.markdown(
        '<div class="brand-mark">B.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-brand">BANK ANALYTICS</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-caption">'
        'Customer intelligence<br>'
        'Retention & predictive analytics'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div style="color:#C5D2E0;font-size:0.75rem;'
        'font-weight:750;letter-spacing:1px;'
        'margin-bottom:0.6rem;">WORKSPACE</div>',
        unsafe_allow_html=True,
    )

    for nav_page in PAGES:
        st.button(
            nav_page,
            key=f"nav_{nav_page}",
            on_click=set_page,
            args=(nav_page,),
            type=(
                "primary"
                if st.session_state["current_page"] == nav_page
                else "secondary"
            ),
            use_container_width=True,
        )

    st.divider()

    st.markdown(
        '<div class="sidebar-caption">'
        'MACHINE LEARNING PORTFOLIO<br>'
        'Customer Churn Analysis'
        '</div>',
        unsafe_allow_html=True,
    )


page = st.session_state["current_page"]


# =========================================================
# PAGE 1 — DASHBOARD
# =========================================================

if page == "Dashboard":
    page_header(
        APP_TITLE,
        "Monitor customer retention, identify churn patterns, "
        "and explore the bank's customer portfolio.",
    )

    total = len(df)
    churned = int(df["Exited"].sum())
    retained = total - churned
    churn_rate = df["Exited"].mean() * 100
    retention_rate = 100 - churn_rate

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", f"{total:,}")
    k2.metric("Churned Customers", f"{churned:,}")
    k3.metric("Retained Customers", f"{retained:,}")
    k4.metric("Churn Rate", f"{churn_rate:.2f}%")

    section("Portfolio overview")

    left, right = st.columns([0.92, 1.35], gap="medium")

    with left:
        status = pd.DataFrame({
            "Status": ["Stayed", "Churned"],
            "Customers": [retained, churned],
        })

        fig = px.pie(
            status,
            names="Status",
            values="Customers",
            hole=0.70,
            color="Status",
            color_discrete_map=CHART_COLORS,
        )

        fig.update_traces(
            textposition="outside",
            texttemplate="%{percent:.1%}",
            hovertemplate=(
                "%{label}<br>Customers: %{value:,}<extra></extra>"
            ),
            marker=dict(line=dict(color=WHITE, width=3)),
            sort=False,
        )

        fig.add_annotation(
            text=(
                f"<b>{retention_rate:.1f}%</b>"
                "<br><span style='font-size:11px'>retained</span>"
            ),
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=19, color=NAVY),
        )

        fig.update_layout(title="Customer retention split")
        show_chart(fig, height=365)

    with right:
        geo = (
            df.groupby("Geography", as_index=False)
            .agg(
                Customers=("Exited", "size"),
                Churn=("Exited", "mean"),
            )
        )

        geo["Churn rate"] = geo["Churn"] * 100
        geo = geo.sort_values("Churn rate", ascending=True)

        fig = px.bar(
            geo,
            x="Churn rate",
            y="Geography",
            orientation="h",
            text=geo["Churn rate"].map(lambda x: f"{x:.1f}%"),
            color="Geography",
            color_discrete_map=CHART_COLORS,
            custom_data=["Customers"],
        )

        fig.update_traces(
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{y}<br>Churn rate: %{x:.2f}%"
                "<br>Customers: %{customdata[0]:,}<extra></extra>"
            ),
        )

        fig.update_layout(
            title="Churn rate by geography",
            showlegend=False,
            xaxis_title="Customers who churned (%)",
            yaxis_title="",
            xaxis=dict(
                range=[0, max(38, geo["Churn rate"].max() + 7)]
            ),
        )

        show_chart(fig, height=365, horizontal=True)

    section("Key findings")

    germany_rate = (
        df.loc[df["Geography"] == "Germany", "Exited"].mean() * 100
    )

    inactive_rate = (
        df.loc[df["IsActiveMember"] == 0, "Exited"].mean() * 100
    )

    active_rate = (
        df.loc[df["IsActiveMember"] == 1, "Exited"].mean() * 100
    )

    f1, f2, f3 = st.columns(3, gap="medium")

    with f1:
        info_card(
            "GEOGRAPHY",
            f"{germany_rate:.1f}%",
            "Observed churn rate in Germany, the highest of the three countries.",
        )

    with f2:
        info_card(
            "CUSTOMER AGE",
            "45 vs 36",
            "Median age of churned customers versus retained customers.",
        )

    with f3:
        info_card(
            "ACTIVITY",
            f"{inactive_rate:.1f}% vs {active_rate:.1f}%",
            "Churn rate among inactive versus active members.",
        )

    st.markdown(
        '<div class="footer-note">Descriptive findings reflect this dataset '
        'and do not establish causation. Small customer groups should be '
        'interpreted cautiously.</div>',
        unsafe_allow_html=True,
    )


# =========================================================
# PAGE 2 — CHURN PREDICTION
# =========================================================

elif page == "Churn Prediction":
    page_header(
        "Churn Prediction",
        "Estimate a customer's churn probability using the trained "
        "Random Forest classification pipeline.",
    )

    st.markdown(
        '<div class="info-card">'
        '<div class="info-label">PREDICTION WORKSPACE</div>'
        '<div class="info-description">Enter customer characteristics '
        'below. The model estimates churn probability; the classification '
        'threshold determines the predicted class.</div></div>',
        unsafe_allow_html=True,
    )

    st.write("")

    with st.form("prediction_form"):
        st.markdown("#### Customer profile")

        left, right = st.columns(2, gap="large")

        with left:
            credit_score = st.number_input(
                "Credit score", 300, 900, 650, step=1
            )

            geography = st.selectbox(
                "Geography", sorted(df["Geography"].dropna().unique())
            )

            gender = st.selectbox(
                "Gender", sorted(df["Gender"].dropna().unique())
            )

            age = st.number_input("Age", 18, 100, 40, step=1)
            tenure = st.number_input(
                "Tenure (years)", 0, 10, 5, step=1
            )

        with right:
            balance = st.number_input(
                "Account balance",
                min_value=0.0,
                value=100000.0,
                step=1000.0,
            )

            products = st.selectbox(
                "Number of products", [1, 2, 3, 4]
            )

            has_card = st.selectbox(
                "Has credit card?", ["Yes", "No"]
            )

            active_member = st.selectbox(
                "Active member?", ["Yes", "No"]
            )

            salary = st.number_input(
                "Estimated salary",
                min_value=0.0,
                value=100000.0,
                step=1000.0,
            )

        submitted = st.form_submit_button(
            "Run churn prediction",
            use_container_width=True,
        )

    if submitted:
        customer = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": int(has_card == "Yes"),
            "IsActiveMember": int(active_member == "Yes"),
            "EstimatedSalary": salary,
        }])

        probability = float(model.predict_proba(customer)[0, 1])
        is_high_risk = probability >= CHURN_THRESHOLD

        st.write("")
        section("Prediction result")

        result_left, result_right = st.columns(
            [1, 1.4], gap="medium"
        )

        with result_left:
            status_text = (
                "HIGHER RISK" if is_high_risk else "LOWER RISK"
            )

            risk_color = RED if is_high_risk else BLUE

            st.markdown(
                f"""
                <div class="risk-panel">
                    <div class="info-label">
                        ESTIMATED CHURN PROBABILITY
                    </div>
                    <div style="font-size:2.6rem;font-weight:800;
                                color:{risk_color};margin:0.5rem 0">
                        {probability:.1%}
                    </div>
                    <div style="display:inline-block;background:{BG};
                                color:{risk_color};padding:6px 10px;
                                border-radius:6px;font-size:0.75rem;
                                font-weight:800;letter-spacing:0.6px">
                        {status_text}
                    </div>
                    <div class="info-description" style="margin-top:1rem">
                        Classification threshold: {CHURN_THRESHOLD:.0%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with result_right:
            st.markdown("#### Probability scale")
            st.progress(float(np.clip(probability, 0, 1)))

            st.caption(
                "The progress bar represents the model's estimated "
                "probability, not certainty."
            )

            if is_high_risk:
                st.error(
                    "The model classifies this customer as higher churn risk."
                )

                st.write(
                    "Consider reviewing customer feedback, engagement, "
                    "and service needs before selecting a retention action."
                )
            else:
                st.success(
                    "The model classifies this customer as lower churn risk."
                )

                st.write(
                    "Maintain engagement and continue monitoring "
                    "customer satisfaction."
                )

        st.caption(
            "This estimate supports decision-making and is not a guarantee "
            "of future customer behaviour."
        )


# =========================================================
# PAGE 3 — MODEL INSIGHTS
# =========================================================

elif page == "Model Insights":
    page_header(
        "Model Insights",
        "Evaluate predictive performance and understand which features "
        "contribute most to the Random Forest model.",
    )

    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"
        ],
        "Logistic Regression": [
            0.8080, 0.5891, 0.1867, 0.2836, 0.7748
        ],
        "Random Forest": [
            0.8400, 0.6154, 0.5700, 0.5918, 0.8508
        ],
    })

    section("Model performance")

    m1, m2, m3 = st.columns(3)
    m1.metric("Random Forest ROC-AUC", "0.8508")
    m2.metric("Random Forest F1-score", "0.5918")
    m3.metric("Random Forest Recall", "57.00%")

    # FIX: Removed unsupported barmode_groupgap argument.
    fig = px.bar(
        metrics,
        x="Metric",
        y=["Logistic Regression", "Random Forest"],
        barmode="group",
        color_discrete_map={
            "Logistic Regression": TEAL,
            "Random Forest": NAVY,
        },
        title="Performance comparison",
        labels={"value": "Score", "variable": "Model"},
        range_y=[0, 1],
    )

    fig.update_traces(
        hovertemplate=(
            "%{x}<br>%{fullData.name}: %{y:.3f}<extra></extra>"
        )
    )

    show_chart(fig, height=390)

    st.dataframe(
        metrics.style.format({
            "Logistic Regression": "{:.4f}",
            "Random Forest": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Logistic Regression used threshold 0.50; Random Forest used "
        "threshold 0.35. Accuracy, precision, recall and F1 are threshold-"
        "dependent, so compare them cautiously. ROC-AUC is threshold-independent."
    )

    section("Feature importance")

    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["model"]

        names = preprocessor.get_feature_names_out()
        values = classifier.feature_importances_

        importance = pd.DataFrame({
            "Feature": names,
            "Importance": values,
        }).sort_values(
            "Importance", ascending=False
        ).head(10)

        importance = importance.sort_values(
            "Importance", ascending=True
        )

        importance["Feature"] = (
            importance["Feature"]
            .str.replace("num__", "", regex=False)
            .str.replace("cat__", "", regex=False)
            .str.replace("_", " ", regex=False)
        )

        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            text=importance["Importance"].map(
                lambda x: f"{x:.3f}"
            ),
            color_discrete_sequence=[BLUE],
            title="Top 10 predictive features",
        )

        fig.update_traces(
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{y}<br>Importance: %{x:.3f}<extra></extra>"
            ),
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="Feature importance",
            yaxis_title="",
            xaxis=dict(
                range=[0, importance["Importance"].max() * 1.3]
            ),
        )

        show_chart(fig, height=440, horizontal=True)

    except Exception as error:
        st.warning(
            f"Could not display feature importance: {error}"
        )

    section("Business recommendations")

    r1, r2 = st.columns(2, gap="medium")

    with r1:
        info_card(
            "01 · RETENTION",
            "Prioritise engagement",
            "Investigate inactive customer groups and test relevant "
            "engagement interventions.",
        )

        st.write("")

        info_card(
            "02 · SEGMENTATION",
            "Review customer groups",
            "Explore observed churn differences by geography, age, "
            "and number of products.",
        )

    with r2:
        info_card(
            "03 · VALIDATION",
            "Measure outcomes",
            "Run controlled retention tests and monitor their results "
            "before scaling interventions.",
        )

        st.write("")

        info_card(
            "04 · GOVERNANCE",
            "Use predictions carefully",
            "Feature importance indicates predictive contribution, "
            "not causation or guaranteed customer behaviour.",
        )


# =========================================================
# PAGE 4 — CUSTOMER ANALYTICS
# =========================================================

elif page == "Customer Analytics":
    page_header(
        "Customer Analytics",
        "Filter customer segments and explore demographic and "
        "behavioural patterns in the dataset.",
    )

    section("Segment filters")

    filter1, filter2 = st.columns(2)

    with filter1:
        geographies = sorted(
            df["Geography"].dropna().unique()
        )

        selected_geographies = st.multiselect(
            "Geography",
            geographies,
            default=geographies,
        )

    with filter2:
        genders = sorted(
            df["Gender"].dropna().unique()
        )

        selected_genders = st.multiselect(
            "Gender",
            genders,
            default=genders,
        )

    filtered = df[
        df["Geography"].isin(selected_geographies)
        & df["Gender"].isin(selected_genders)
    ].copy()

    if filtered.empty:
        st.warning("No customers match the selected filters.")
        st.stop()

    churn_rate = filtered["Exited"].mean() * 100
    churned = int(filtered["Exited"].sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers in Segment", f"{len(filtered):,}")
    k2.metric("Churned Customers", f"{churned:,}")
    k3.metric("Segment Churn Rate", f"{churn_rate:.2f}%")
    k4.metric("Median Age", f"{filtered['Age'].median():.0f}")

    section("Customer behaviour")

    left, right = st.columns(2, gap="medium")

    with left:
        age_data = filtered.copy()

        age_data["Status"] = age_data["Exited"].map({
            0: "Stayed",
            1: "Churned",
        })

        fig = px.histogram(
            age_data,
            x="Age",
            color="Status",
            barmode="overlay",
            opacity=0.78,
            nbins=25,
            color_discrete_map=CHART_COLORS,
            title="Age distribution by customer status",
        )

        fig.update_layout(
            xaxis_title="Age",
            yaxis_title="Number of customers",
        )

        show_chart(fig, height=380)

    with right:
        products_data = (
            filtered.groupby("NumOfProducts", as_index=False)
            .agg(
                Customers=("Exited", "size"),
                Churn_Rate=("Exited", "mean"),
            )
        )

        products_data["Churn rate"] = (
            products_data["Churn_Rate"] * 100
        )

        fig = px.bar(
            products_data,
            x="NumOfProducts",
            y="Churn rate",
            text=products_data["Churn rate"].map(
                lambda x: f"{x:.1f}%"
            ),
            color_discrete_sequence=[BLUE],
            title="Churn rate by number of products",
            custom_data=["Customers"],
        )

        fig.update_traces(
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "Products: %{x}<br>Churn rate: %{y:.1f}%"
                "<br>Customers: %{customdata[0]:,}<extra></extra>"
            ),
        )

        fig.update_layout(
            xaxis_title="Number of products",
            yaxis_title="Customers who churned (%)",
            yaxis=dict(
                range=[
                    0,
                    max(110, products_data["Churn rate"].max() + 10),
                ]
            ),
        )

        show_chart(fig, height=380)

    section("Activity and churn")

    activity = (
        filtered.groupby("IsActiveMember", as_index=False)
        .agg(
            Customers=("Exited", "size"),
            Churn_Rate=("Exited", "mean"),
        )
    )

    activity["Activity"] = activity["IsActiveMember"].map({
        0: "Inactive",
        1: "Active",
    })

    activity["Churn rate"] = activity["Churn_Rate"] * 100

    fig = px.bar(
        activity,
        x="Activity",
        y="Churn rate",
        text=activity["Churn rate"].map(
            lambda x: f"{x:.1f}%"
        ),
        color="Activity",
        color_discrete_map={
            "Active": TEAL,
            "Inactive": NAVY,
        },
        custom_data=["Customers"],
        title="Churn rate by activity status",
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "%{x}<br>Churn rate: %{y:.1f}%"
            "<br>Customers: %{customdata[0]:,}<extra></extra>"
        ),
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title="Membership activity",
        yaxis_title="Customers who churned (%)",
        yaxis=dict(
            range=[
                0,
                max(35, activity["Churn rate"].max() + 8),
            ]
        ),
    )

    show_chart(fig, height=350)

    section("Customer records")

    display_columns = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Balance",
        "NumOfProducts",
        "IsActiveMember",
        "Exited",
    ]

    records = filtered[display_columns].head(100).copy()

    records["Customer Status"] = records["Exited"].map({
        0: "Stayed",
        1: "Churned",
    })

    records = records.drop(columns=["Exited"])

    st.dataframe(
        records,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Showing {len(records):,} records at most. "
        "Segment statistics describe the filtered dataset."
    )
