import os
import sys

import streamlit as st


# ============================================================
# PROJECT ROOT
# ============================================================

CURRENT_DIRECTORY = os.path.dirname(
    os.path.abspath(
        __file__
    )
)

PROJECT_ROOT = os.path.dirname(
    CURRENT_DIRECTORY
)

if PROJECT_ROOT not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# IMPORTS
# ============================================================

from dashboard.data_loader import (
    load_customer_intelligence,
    filter_dashboard_data,
    get_unique_values
)

from dashboard.ui import (
    configure_plotly_theme,
    apply_dashboard_style,
    render_dashboard_header,
    render_footer,
    render_data_status,
    render_global_download,
    render_data_error,
    render_filter_summary
)

from dashboard.executive_overview import (
    ExecutiveOverview
)

from dashboard.churn_intelligence import (
    ChurnIntelligence
)

from dashboard.segmentation_intelligence import (
    SegmentationIntelligence
)

from dashboard.clv_intelligence import (
    CLVIntelligence
)

from dashboard.recommendation_intelligence import (
    RecommendationIntelligence
)

from dashboard.customer_explorer import (
    CustomerExplorer
)

from dashboard.explainability_intelligence import (
    ExplainabilityIntelligence
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=
        "AI Customer Intelligence Platform",

    page_icon=
        "🚀",

    layout=
        "wide",

    initial_sidebar_state=
        "expanded"
)


# ============================================================
# GLOBAL THEME
# ============================================================

configure_plotly_theme()

apply_dashboard_style()


# ============================================================
# DATA PATH
# ============================================================

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "dashboard",
    "unified_customer_intelligence.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = (
        load_customer_intelligence(
            DATA_PATH
        )
    )

except FileNotFoundError as error:

    st.error(
        "Customer intelligence dataset "
        "has not been generated yet."
    )

    st.info(
        "Run:\n\n"
        "`python generate_all_customer_intelligence.py`"
    )

    render_data_error(
        "Dataset loading failed.",
        error
    )

    st.stop()

except Exception as error:

    render_data_error(
        (
            "Unable to load the customer "
            "intelligence dataset."
        ),
        error
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

render_dashboard_header()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🚀 Customer Intelligence"
)

st.sidebar.caption(
    "AI-powered customer analytics"
)


# ============================================================
# NAVIGATION
# ============================================================

page = (
    st.sidebar.radio(
        "Navigation",

        options=[
            "🏠 Executive Overview",
            "👤 Customer Explorer",
            "⚠️ Churn Intelligence",
            "🧩 Customer Segmentation",
            "💰 CLV Intelligence",
            "🎯 Recommendation Intelligence",
            "🧠 Explainability Intelligence"
        ]
    )
)


# ============================================================
# GLOBAL FILTERS
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "🔎 Global Filters"
)


# ------------------------------------------------------------
# CUSTOMER SEGMENT
# ------------------------------------------------------------

segment_options = (
    get_unique_values(
        df,
        "Customer Segment"
    )
)

selected_segments = (
    st.sidebar.multiselect(
        "Customer Segment",

        options=
            segment_options,

        default=
            segment_options
    )
)


# ------------------------------------------------------------
# CHURN RISK
# ------------------------------------------------------------

churn_risk_options = (
    get_unique_values(
        df,
        "Churn Risk"
    )
)

selected_churn_risks = (
    st.sidebar.multiselect(
        "Churn Risk",

        options=
            churn_risk_options,

        default=
            churn_risk_options
    )
)


# ------------------------------------------------------------
# CLV VALUE BAND
# ------------------------------------------------------------

clv_options = (
    get_unique_values(
        df,
        "CLV Value Band"
    )
)

selected_clv_bands = (
    st.sidebar.multiselect(
        "CLV Value Band",

        options=
            clv_options,

        default=
            clv_options
    )
)


# ------------------------------------------------------------
# CUSTOMER SEARCH
# ------------------------------------------------------------

customer_search = (
    st.sidebar.text_input(
        "Customer ID",

        placeholder=
            "Search Customer ID"
    )
)


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = (
    filter_dashboard_data(
        df=
            df,

        segments=
            selected_segments,

        churn_risks=
            selected_churn_risks,

        clv_bands=
            selected_clv_bands,

        customer_search=
            customer_search
    )
)


# ============================================================
# FILTER SUMMARY
# ============================================================

render_filter_summary(
    original_df=
        df,

    filtered_df=
        filtered_df
)


# ============================================================
# DATA STATUS
# ============================================================

render_data_status(
    df=
        df,

    file_path=
        DATA_PATH
)


# ============================================================
# EXPORT
# ============================================================

render_global_download(
    filtered_df
)


# ============================================================
# EMPTY FILTER RESULT
# ============================================================

if filtered_df.empty:

    st.warning(
        "No customers match the selected filters."
    )

    st.info(
        "Adjust Customer Segment, Churn Risk, "
        "CLV Value Band or Customer ID."
    )

    render_footer()

    st.stop()


# ============================================================
# PAGE ROUTING
# ============================================================

try:

    # --------------------------------------------------------
    # EXECUTIVE OVERVIEW
    # --------------------------------------------------------

    if page == "🏠 Executive Overview":

        dashboard = (
            ExecutiveOverview()
        )

        dashboard.render(
            filtered_df
        )

    # --------------------------------------------------------
    # CUSTOMER EXPLORER
    # --------------------------------------------------------

    elif page == "👤 Customer Explorer":

        dashboard = (
            CustomerExplorer()
        )

        dashboard.render(
            filtered_df
        )

    # --------------------------------------------------------
    # CHURN
    # --------------------------------------------------------

    elif page == "⚠️ Churn Intelligence":

        dashboard = (
            ChurnIntelligence()
        )

        dashboard.render(
            filtered_df
        )

    # --------------------------------------------------------
    # SEGMENTATION
    # --------------------------------------------------------

    elif page == "🧩 Customer Segmentation":

        dashboard = (
            SegmentationIntelligence()
        )

        dashboard.render(
            filtered_df
        )

    # --------------------------------------------------------
    # CLV
    # --------------------------------------------------------

    elif page == "💰 CLV Intelligence":

        dashboard = (
            CLVIntelligence()
        )

        dashboard.render(
            filtered_df
        )

    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

    elif page == "🎯 Recommendation Intelligence":

        dashboard = (
            RecommendationIntelligence()
        )

        dashboard.render(
            filtered_df
        )

    # --------------------------------------------------------
    # EXPLAINABILITY
    # --------------------------------------------------------

    elif page == "🧠 Explainability Intelligence":

        dashboard = (
            ExplainabilityIntelligence(
                project_root=
                    PROJECT_ROOT
            )
        )

        dashboard.render(
            filtered_df
        )


except Exception as error:

    render_data_error(
        (
            f"An error occurred while "
            f"rendering {page}."
        ),
        error
    )


# ============================================================
# FOOTER
# ============================================================

render_footer()