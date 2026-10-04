import os
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.io as pio


CUSTOMER_ID = "Customer ID"

def configure_plotly_theme():

    template = pio.templates["plotly_white"]

    template.layout.font = {
        "family": "Inter, Segoe UI, Arial, sans-serif",
        "size": 13,
        "color": "#334155"
    }

    template.layout.title = {
        "font": {
            "family": "Inter, Segoe UI, Arial, sans-serif",
            "size": 19,
            "color": "#0f172a"
        },
        "x": 0.02,
        "xanchor": "left"
    }

    template.layout.paper_bgcolor = "rgba(0,0,0,0)"
    template.layout.plot_bgcolor = "rgba(255,255,255,0)"

    template.layout.margin = {
        "l": 35,
        "r": 25,
        "t": 65,
        "b": 45
    }

    template.layout.hoverlabel = {
        "bgcolor": "#111827",
        "font": {
            "color": "white",
            "size": 12
        },
        "bordercolor": "#111827"
    }

    template.layout.legend = {
        "bgcolor": "rgba(255,255,255,0)",
        "font": {
            "size": 12
        }
    }

    template.layout.xaxis = {
        "gridcolor": "#eef2f7",
        "linecolor": "#cbd5e1",
        "zerolinecolor": "#e2e8f0"
    }

    template.layout.yaxis = {
        "gridcolor": "#eef2f7",
        "linecolor": "#cbd5e1",
        "zerolinecolor": "#e2e8f0"
    }

    pio.templates["cip_theme"] = template

    pio.templates.default = "cip_theme"

# ============================================================
# GLOBAL DASHBOARD STYLE
# ============================================================

def apply_dashboard_style():

    st.markdown(
        """
<style>

/* ==========================================================
   MAIN PAGE
   ========================================================== */

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1550px;
}


/* ==========================================================
   APP BACKGROUND
   ========================================================== */

.stApp {
    background:
        linear-gradient(
            135deg,
            #f8fafc 0%,
            #eef2ff 50%,
            #f0fdfa 100%
        );
}


/* ==========================================================
   SIDEBAR
   ========================================================== */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #111827 0%,
            #1e1b4b 55%,
            #312e81 100%
        );
}

[data-testid="stSidebar"] * {
    color: white;
}

[data-testid="stSidebar"] label {
    color: white !important;
}

[data-testid="stSidebar"] p {
    color: #e5e7eb;
}


/* ==========================================================
   SIDEBAR INPUTS
   ========================================================== */

[data-testid="stSidebar"] input {
    color: #111827 !important;
    background: white !important;
}

[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: white;
    color: #111827;
}


/* ==========================================================
   MAIN DASHBOARD HEADER
   ========================================================== */

.cip-header {

    background:
        linear-gradient(
            120deg,
            #312e81 0%,
            #4f46e5 35%,
            #7c3aed 65%,
            #0891b2 100%
        );

    padding: 30px 34px;

    border-radius: 22px;

    margin-bottom: 28px;

    box-shadow:
        0 15px 40px
        rgba(79, 70, 229, 0.25);

    color: white;
}


.cip-title {

    font-size: 2.25rem;

    font-weight: 800;

    margin: 0;

    color: white;

    letter-spacing: -0.5px;
}


.cip-subtitle {

    font-size: 1rem;

    margin-top: 10px;

    color: #e0e7ff;
}


.cip-badge {

    display: inline-block;

    margin-top: 18px;

    padding: 7px 15px;

    border-radius: 25px;

    background:
        rgba(
            255,
            255,
            255,
            0.18
        );

    border:
        1px solid
        rgba(
            255,
            255,
            255,
            0.35
        );

    color: white;

    font-size: 0.85rem;

    font-weight: 600;
}


/* ==========================================================
   STREAMLIT METRIC CARDS
   ========================================================== */

div[data-testid="stMetric"] {

    background:
        linear-gradient(
            145deg,
            #ffffff,
            #f8fafc
        );

    border:
        1px solid
        rgba(
            99,
            102,
            241,
            0.15
        );

    border-radius: 18px;

    padding: 20px;

    box-shadow:
        0 8px 25px
        rgba(
            15,
            23,
            42,
            0.06
        );

    transition:
        all 0.25s ease;
}


div[data-testid="stMetric"]:hover {

    transform:
        translateY(-4px);

    box-shadow:
        0 15px 35px
        rgba(
            79,
            70,
            229,
            0.15
        );

    border:
        1px solid
        rgba(
            79,
            70,
            229,
            0.35
        );
}


/* ==========================================================
   METRIC LABEL
   ========================================================== */

div[data-testid="stMetricLabel"] {

    font-size: 0.95rem;

    font-weight: 600;

    color: #475569;
}


/* ==========================================================
   METRIC VALUE
   ========================================================== */

div[data-testid="stMetricValue"] {

    font-size: 2rem;

    font-weight: 800;

    color: #1e1b4b;
}


/* ==========================================================
   HEADINGS
   ========================================================== */

h1 {

    color: #0f172a;

    font-weight: 800;
}


h2,
h3 {

    color: #1e293b;

    font-weight: 700;
}


/* ==========================================================
   DATAFRAME
   ========================================================== */

div[data-testid="stDataFrame"] {

    border-radius: 15px;

    overflow: hidden;

    border:
        1px solid
        #e2e8f0;

    box-shadow:
        0 5px 18px
        rgba(
            15,
            23,
            42,
            0.05
        );
}


/* ==========================================================
   BUTTON
   ========================================================== */

.stDownloadButton button {

    width: 100%;

    border-radius: 10px;

    background:
        linear-gradient(
            90deg,
            #4f46e5,
            #7c3aed
        );

    color: white;

    border: none;

    font-weight: 600;
}


.stDownloadButton button:hover {

    background:
        linear-gradient(
            90deg,
            #4338ca,
            #6d28d9
        );

    color: white;

    border: none;
}


/* ==========================================================
   INFO / WARNING BOXES
   ========================================================== */

div[data-testid="stAlert"] {

    border-radius: 14px;
}


/* ==========================================================
   TABS
   ========================================================== */

button[data-baseweb="tab"] {

    font-weight: 600;
}


/* ==========================================================
   FOOTER
   ========================================================== */

.cip-footer {

    margin-top: 50px;

    padding-top: 20px;

    padding-bottom: 10px;

    border-top:
        1px solid
        #e2e8f0;

    text-align: center;

    font-size: 0.85rem;

    color: #64748b;
}


/* ==========================================================
   CUSTOM COLOR CARDS
   ========================================================== */

.dashboard-card {

    padding: 20px;

    border-radius: 18px;

    color: white;

    box-shadow:
        0 10px 30px
        rgba(
            0,
            0,
            0,
            0.10
        );
}


.card-blue {

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #4f46e5
        );
}


.card-red {

    background:
        linear-gradient(
            135deg,
            #dc2626,
            #f97316
        );
}


.card-purple {

    background:
        linear-gradient(
            135deg,
            #7c3aed,
            #c026d3
        );
}


.card-green {

    background:
        linear-gradient(
            135deg,
            #059669,
            #0d9488
        );
}


.card-title {

    font-size: 0.9rem;

    opacity: 0.88;

    margin-bottom: 8px;
}


.card-value {

    font-size: 2rem;

    font-weight: 800;
}


/* ==========================================================
   RESPONSIVE
   ========================================================== */

@media (
    max-width: 900px
) {

    .cip-title {
        font-size: 1.65rem;
    }

    .cip-header {
        padding: 22px;
    }
}

</style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

def render_dashboard_header():

    # IMPORTANT:
    # HTML is intentionally not indented.
    # Indented HTML can be interpreted as Markdown code.

    header_html = (
        '<div class="cip-header">'
        '<div class="cip-title">'
        '🚀 AI-Powered Customer Intelligence Platform'
        '</div>'
        '<div class="cip-subtitle">'
        'Churn Prediction &nbsp;•&nbsp; '
        'Customer Segmentation &nbsp;•&nbsp; '
        'Predicted 90-Day Revenue &nbsp;•&nbsp; '
        'Product Recommendation'
        '</div>'
        '<div class="cip-badge">'
        '✨ Unified Customer Intelligence Dashboard'
        '</div>'
        '</div>'
    )

    st.markdown(
        header_html,
        unsafe_allow_html=True
    )


# ============================================================
# CUSTOM KPI CARD
# ============================================================

def render_color_card(
    title: str,
    value,
    icon: str = "📊",
    card_class: str = "card-blue"
):

    card_html = (
        f'<div class="dashboard-card {card_class}">'
        f'<div class="card-title">'
        f'{icon} {title}'
        f'</div>'
        f'<div class="card-value">'
        f'{value}'
        f'</div>'
        f'</div>'
    )

    st.markdown(
        card_html,
        unsafe_allow_html=True
    )


# ============================================================
# RISK BADGE
# ============================================================

def format_risk_badge(
    value
):

    risk = str(
        value
    ).strip().lower()

    if risk == "high":

        return "🔴 High"

    if risk == "medium":

        return "🟠 Medium"

    if risk == "low":

        return "🟢 Low"

    return "⚪ Unknown"


# ============================================================
# CLV BADGE
# ============================================================

def format_clv_badge(
    value
):

    band = str(
        value
    ).strip().lower()

    if band == "high":

        return "🟣 High"

    if band == "medium":

        return "🔵 Medium"

    if band == "low":

        return "⚪ Low"

    return "⚫ Unknown"


# ============================================================
# PRIORITY BADGE
# ============================================================

def format_priority_badge(
    value
):

    priority = str(
        value
    ).strip().lower()

    if priority == "critical":

        return "🔥 Critical"

    if priority == "high":

        return "🔴 High"

    if priority == "medium":

        return "🟠 Medium"

    if priority == "low":

        return "🟢 Low"

    return "⚪ Unknown"


# ============================================================
# FOOTER
# ============================================================

def render_footer():

    footer_html = (
        '<div class="cip-footer">'
        'AI-Powered Customer Intelligence Platform'
        '<br>'
        'Machine Learning • Customer Analytics • '
        'Decision Intelligence'
        '</div>'
    )

    st.markdown(
        footer_html,
        unsafe_allow_html=True
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

def get_dataset_information(
    df: pd.DataFrame,
    file_path: str
) -> dict:

    if (
        CUSTOMER_ID
        in df.columns
    ):

        total_customers = int(
            df[
                CUSTOMER_ID
            ]
            .nunique()
        )

    else:

        total_customers = 0

    if os.path.exists(
        file_path
    ):

        timestamp = (
            os.path.getmtime(
                file_path
            )
        )

        modified_time = (
            datetime.fromtimestamp(
                timestamp
            )
        )

        last_updated = (
            modified_time.strftime(
                "%d %b %Y, %I:%M %p"
            )
        )

        file_size_mb = (
            os.path.getsize(
                file_path
            )
            /
            (
                1024
                *
                1024
            )
        )

    else:

        last_updated = "Unknown"

        file_size_mb = 0.0

    return {

        "rows":
            int(
                len(
                    df
                )
            ),

        "columns":
            int(
                len(
                    df.columns
                )
            ),

        "customers":
            total_customers,

        "last_updated":
            last_updated,

        "file_size_mb":
            float(
                file_size_mb
            )
    }


# ============================================================
# DATA STATUS
# ============================================================

def render_data_status(
    df: pd.DataFrame,
    file_path: str
):

    info = (
        get_dataset_information(
            df,
            file_path
        )
    )

    st.sidebar.divider()

    st.sidebar.subheader(
        "📊 Data Status"
    )

    st.sidebar.write(
        f"👥 **Customers:** "
        f"{info['customers']:,}"
    )

    st.sidebar.write(
        f"📄 **Rows:** "
        f"{info['rows']:,}"
    )

    st.sidebar.write(
        f"🧩 **Features:** "
        f"{info['columns']:,}"
    )

    st.sidebar.caption(
        f"Updated: "
        f"{info['last_updated']}"
    )


# ============================================================
# CSV EXPORT
# ============================================================

def dataframe_to_csv_bytes(
    df: pd.DataFrame
) -> bytes:

    return (
        df
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )


# ============================================================
# GLOBAL DOWNLOAD
# ============================================================

def render_global_download(
    df: pd.DataFrame
):

    st.sidebar.divider()

    st.sidebar.subheader(
        "📥 Export"
    )

    data = (
        dataframe_to_csv_bytes(
            df
        )
    )

    st.sidebar.download_button(
        label=
            "Download Filtered Customers",

        data=
            data,

        file_name=
            "filtered_customer_intelligence.csv",

        mime=
            "text/csv",

        use_container_width=
            True
    )


# ============================================================
# ERROR
# ============================================================

def render_data_error(
    title: str,
    error
):

    st.error(
        title
    )

    with st.expander(
        "Technical Details"
    ):

        st.exception(
            error
        )


# ============================================================
# FILTER SUMMARY
# ============================================================

def render_filter_summary(
    original_df: pd.DataFrame,
    filtered_df: pd.DataFrame
):

    original_customers = int(
        original_df[
            CUSTOMER_ID
        ]
        .nunique()
    )

    filtered_customers = int(
        filtered_df[
            CUSTOMER_ID
        ]
        .nunique()
    )

    if original_customers == 0:

        percentage = 0.0

    else:

        percentage = (
            filtered_customers
            /
            original_customers
            *
            100
        )

    st.sidebar.caption(
        f"Showing "
        f"{filtered_customers:,} / "
        f"{original_customers:,} customers "
        f"({percentage:.1f}%)"
    )