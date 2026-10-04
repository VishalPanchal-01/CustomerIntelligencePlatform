import pandas as pd
import streamlit as st


# =============================================================
# PAGE HEADER
# =============================================================

def render_page_header(
    title: str,
    description: str = None
):

    st.title(
        title
    )

    if description:

        st.caption(
            description
        )


# =============================================================
# KPI CARDS
# =============================================================

def render_overview_kpis(
    metrics: dict
):

    col1, col2, col3, col4 = (
        st.columns(
            4
        )
    )

    with col1:

        st.metric(
            label=
                "Total Customers",

            value=
                f"{metrics['total_customers']:,}"
        )

    with col2:

        st.metric(
            label=
                "High Churn Risk",

            value=
                f"{metrics['high_risk_customers']:,}"
        )

    with col3:

        st.metric(
            label=
                "High Value Customers",

            value=
                f"{metrics['high_value_customers']:,}"
        )

    with col4:

        st.metric(
            label=
                "Avg. Predicted 90-Day Revenue",

            value=
                f"{metrics['average_predicted_revenue']:,.2f}"
        )


# =============================================================
# SECONDARY KPI CARDS
# =============================================================

def render_secondary_kpis(
    metrics: dict
):

    col1, col2 = (
        st.columns(
            2
        )
    )

    with col1:

        st.metric(
            label=
                "Total Predicted 90-Day Revenue",

            value=
                f"{metrics['total_predicted_revenue']:,.2f}"
        )

    with col2:

        st.metric(
            label=
                "Average Churn Probability",

            value=
                (
                    f"{metrics['average_churn_probability'] * 100:.2f}%"
                )
        )


# =============================================================
# DATASET SUMMARY
# =============================================================

def render_dataset_summary(
    df: pd.DataFrame
):

    st.subheader(
        "Customer Intelligence Dataset"
    )

    st.write(
        f"Customers displayed: "
        f"**{df['Customer ID'].nunique():,}**"
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# =============================================================
# CUSTOMER PROFILE
# =============================================================

def render_customer_profile(
    customer: pd.Series
):

    st.subheader(
        "Customer Profile"
    )

    st.write(
        f"**Customer ID:** "
        f"{customer.get('Customer ID', '')}"
    )

    col1, col2, col3 = (
        st.columns(
            3
        )
    )

    with col1:

        st.metric(
            "Customer Segment",
            customer.get(
                "Customer Segment",
                "Not Available"
            )
        )

    with col2:

        churn_probability = (
            customer.get(
                "Churn Probability"
            )
        )

        if pd.notna(
            churn_probability
        ):

            churn_value = (
                f"{float(churn_probability) * 100:.2f}%"
            )

        else:

            churn_value = (
                "Not Available"
            )

        st.metric(
            "Churn Probability",
            churn_value
        )

    with col3:

        st.metric(
            "Churn Risk",
            customer.get(
                "Churn Risk",
                "Not Available"
            )
        )

    col4, col5, col6 = (
        st.columns(
            3
        )
    )

    with col4:

        revenue = (
            customer.get(
                "Predicted 90-Day Revenue"
            )
        )

        if pd.notna(
            revenue
        ):

            revenue_text = (
                f"{float(revenue):,.2f}"
            )

        else:

            revenue_text = (
                "Not Available"
            )

        st.metric(
            "Predicted 90-Day Revenue",
            revenue_text
        )

    with col5:

        st.metric(
            "CLV Value Band",
            customer.get(
                "CLV Value Band",
                "Not Available"
            )
        )

    with col6:

        st.metric(
            "Top Recommended Product",
            customer.get(
                "Top Recommended Product",
                "Not Available"
            )
        )


# =============================================================
# CUSTOMER BEHAVIOR
# =============================================================

def render_customer_behavior(
    customer: pd.Series
):

    st.subheader(
        "Customer Behaviour"
    )

    columns = (
        st.columns(
            6
        )
    )

    features = [
        (
            "Recency",
            "Recency"
        ),

        (
            "Frequency",
            "Frequency"
        ),

        (
            "Monetary",
            "Monetary"
        ),

        (
            "Total Items",
            "TotalItems"
        ),

        (
            "Average Order Value",
            "AverageOrderValue"
        ),

        (
            "Tenure",
            "Tenure"
        )
    ]

    for container, (
        label,
        column
    ) in zip(
        columns,
        features
    ):

        value = (
            customer.get(
                column
            )
        )

        with container:

            if pd.isna(
                value
            ):

                display_value = (
                    "N/A"
                )

            elif isinstance(
                value,
                float
            ):

                display_value = (
                    f"{value:,.2f}"
                )

            else:

                display_value = (
                    str(
                        value
                    )
                )

            st.metric(
                label,
                display_value
            )


# =============================================================
# RECOMMENDATION DETAILS
# =============================================================

def render_customer_recommendations(
    customer: pd.Series
):

    st.subheader(
        "Recommended Products"
    )

    top_product = (
        customer.get(
            "Top Recommended Product"
        )
    )

    top_code = (
        customer.get(
            "Top Recommended Stock Code"
        )
    )

    score = (
        customer.get(
            "Top Recommendation Score"
        )
    )

    source = (
        customer.get(
            "Recommendation Source"
        )
    )

    if pd.notna(
        top_product
    ):

        st.write(
            f"**Top Recommendation:** "
            f"{top_product}"
        )

    if pd.notna(
        top_code
    ):

        st.write(
            f"**Stock Code:** "
            f"{top_code}"
        )

    if pd.notna(
        score
    ):

        st.write(
            f"**Recommendation Score:** "
            f"{float(score):.4f}"
        )

    if pd.notna(
        source
    ):

        st.write(
            f"**Recommendation Source:** "
            f"{source}"
        )

    recommended_products = (
        customer.get(
            "Recommended Products"
        )
    )

    if pd.notna(
        recommended_products
    ):

        products = [
            product.strip()
            for product
            in str(
                recommended_products
            ).split(
                "|"
            )
            if product.strip()
        ]

        if products:

            recommendation_table = (
                pd.DataFrame(
                    {
                        "Rank":
                            range(
                                1,
                                len(products) + 1
                            ),

                        "Product":
                            products
                    }
                )
            )

            st.dataframe(
                recommendation_table,
                use_container_width=True,
                hide_index=True
            )

    else:

        st.info(
            "No recommendations available "
            "for this customer."
        )