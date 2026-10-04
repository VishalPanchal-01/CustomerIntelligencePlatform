import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.executive_overview import (
    ExecutiveOverview
)

from dashboard.churn_intelligence import (
    ChurnIntelligence
)

from dashboard.recommendation_intelligence import (
    RecommendationIntelligence
)


CUSTOMER_ID = "Customer ID"


class CustomerExplorer:

    BEHAVIOR_FEATURES = [
        "Recency",
        "Frequency",
        "Monetary",
        "TotalItems",
        "AverageOrderValue",
        "Tenure"
    ]

    # =========================================================
    # PREPARE DATA
    # =========================================================

    def prepare_data(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = df.copy()

        if CUSTOMER_ID not in data.columns:

            raise ValueError(
                "Customer ID column not found."
            )

        numeric_columns = [
            "Churn Probability",
            "Predicted 90-Day Revenue",
            "Top Recommendation Score",
            "Recency",
            "Frequency",
            "Monetary",
            "TotalItems",
            "AverageOrderValue",
            "Tenure"
        ]

        for column in numeric_columns:

            if column in data.columns:

                data[
                    column
                ] = pd.to_numeric(
                    data[
                        column
                    ],
                    errors="coerce"
                )

        return data

    # =========================================================
    # GET CUSTOMER
    # =========================================================

    def get_customer(
        self,
        df: pd.DataFrame,
        customer_id
    ):

        data = (
            self.prepare_data(
                df
            )
        )

        customer_rows = (
            data[
                data[
                    CUSTOMER_ID
                ]
                ==
                customer_id
            ]
        )

        if customer_rows.empty:

            return None

        return (
            customer_rows
            .iloc[0]
        )

    # =========================================================
    # CUSTOMER BUSINESS PRIORITY
    # =========================================================

    def customer_priority(
        self,
        df: pd.DataFrame,
        customer_id
    ) -> dict:

        data = (
            self.prepare_data(
                df
            )
        )

        customer_rows = (
            data[
                data[
                    CUSTOMER_ID
                ]
                ==
                customer_id
            ]
        )

        if customer_rows.empty:

            return {
                "priority":
                    "Unknown",

                "business_action":
                    "Customer not found."
            }

        overview = (
            ExecutiveOverview()
        )

        result = (
            overview
            .add_customer_priority(
                customer_rows
            )
            .iloc[0]
        )

        return {

            "priority":
                result[
                    "Customer Priority"
                ],

            "business_action":
                result[
                    "Recommended Business Action"
                ]
        }

    # =========================================================
    # RETENTION PRIORITY SCORE
    # =========================================================

    def retention_priority_score(
        self,
        df: pd.DataFrame,
        customer_id
    ):

        churn = (
            ChurnIntelligence()
        )

        ranking = (
            churn
            .retention_priority_customers(
                df,
                top_n=
                    len(df)
            )
        )

        if ranking.empty:

            return None

        customer = (
            ranking[
                ranking[
                    CUSTOMER_ID
                ]
                ==
                customer_id
            ]
        )

        if customer.empty:

            return None

        return float(
            customer[
                "Retention Priority Score"
            ]
            .iloc[0]
        )

    # =========================================================
    # RETENTION RANK
    # =========================================================

    def retention_rank(
        self,
        df: pd.DataFrame,
        customer_id
    ):

        churn = (
            ChurnIntelligence()
        )

        ranking = (
            churn
            .retention_priority_customers(
                df,
                top_n=
                    len(df)
            )
        )

        if ranking.empty:

            return None

        ranking = (
            ranking
            .reset_index(
                drop=True
            )
        )

        ranking[
            "Retention Rank"
        ] = (
            np.arange(
                1,
                len(ranking) + 1
            )
        )

        customer = (
            ranking[
                ranking[
                    CUSTOMER_ID
                ]
                ==
                customer_id
            ]
        )

        if customer.empty:

            return None

        return int(
            customer[
                "Retention Rank"
            ]
            .iloc[0]
        )

    # =========================================================
    # CUSTOMER BEHAVIOR PERCENTILES
    # =========================================================

    def behavior_percentiles(
        self,
        df: pd.DataFrame,
        customer_id
    ) -> pd.DataFrame:

        data = (
            self.prepare_data(
                df
            )
        )

        customer = (
            self.get_customer(
                data,
                customer_id
            )
        )

        if customer is None:

            return pd.DataFrame()

        rows = []

        for feature in (
            self.BEHAVIOR_FEATURES
        ):

            if feature not in data.columns:

                continue

            series = (
                data[
                    feature
                ]
                .dropna()
            )

            customer_value = (
                customer.get(
                    feature
                )
            )

            if (
                series.empty
                or
                pd.isna(
                    customer_value
                )
            ):

                continue

            percentile = float(
                (
                    series
                    <=
                    customer_value
                )
                .mean()
                *
                100
            )

            rows.append(
                {
                    "Feature":
                        feature,

                    "Customer Value":
                        float(
                            customer_value
                        ),

                    "Population Average":
                        float(
                            series.mean()
                        ),

                    "Percentile":
                        percentile
                }
            )

        return pd.DataFrame(
            rows
        )

    # =========================================================
    # RADAR PROFILE DATA
    # =========================================================

    def normalized_behavior_profile(
        self,
        df: pd.DataFrame,
        customer_id
    ) -> pd.DataFrame:

        percentiles = (
            self.behavior_percentiles(
                df,
                customer_id
            )
        )

        if percentiles.empty:

            return pd.DataFrame()

        profile = (
            percentiles[
                [
                    "Feature",
                    "Percentile"
                ]
            ]
            .copy()
        )

        # -----------------------------------------------------
        # Recency interpretation is reversed.
        #
        # Lower recency = more recently active.
        # For visualization we convert to:
        # Customer Engagement Recency Score
        # -----------------------------------------------------

        recency_mask = (
            profile[
                "Feature"
            ]
            ==
            "Recency"
        )

        profile.loc[
            recency_mask,
            "Percentile"
        ] = (
            100
            -
            profile.loc[
                recency_mask,
                "Percentile"
            ]
        )

        profile.loc[
            recency_mask,
            "Feature"
        ] = (
            "Recency Engagement"
        )

        return profile

    # =========================================================
    # RECOMMENDATION LIST
    # =========================================================

    def recommendations(
        self,
        df: pd.DataFrame,
        customer_id
    ) -> pd.DataFrame:

        recommender = (
            RecommendationIntelligence()
        )

        return (
            recommender
            .customer_recommendations(
                df,
                customer_id
            )
        )

    # =========================================================
    # CUSTOMER REPORT DATA
    # =========================================================

    def build_customer_report(
        self,
        df: pd.DataFrame,
        customer_id
    ) -> pd.DataFrame:

        customer = (
            self.get_customer(
                df,
                customer_id
            )
        )

        if customer is None:

            return pd.DataFrame()

        priority = (
            self.customer_priority(
                df,
                customer_id
            )
        )

        retention_score = (
            self.retention_priority_score(
                df,
                customer_id
            )
        )

        retention_rank = (
            self.retention_rank(
                df,
                customer_id
            )
        )

        report_rows = [

            {
                "Category":
                    "Identity",

                "Metric":
                    "Customer ID",

                "Value":
                    customer.get(
                        CUSTOMER_ID,
                        ""
                    )
            },

            {
                "Category":
                    "Segmentation",

                "Metric":
                    "Customer Segment",

                "Value":
                    customer.get(
                        "Customer Segment",
                        "Not Available"
                    )
            },

            {
                "Category":
                    "Churn",

                "Metric":
                    "Churn Probability",

                "Value":
                    customer.get(
                        "Churn Probability",
                        "Not Available"
                    )
            },

            {
                "Category":
                    "Churn",

                "Metric":
                    "Churn Risk",

                "Value":
                    customer.get(
                        "Churn Risk",
                        "Not Available"
                    )
            },

            {
                "Category":
                    "CLV",

                "Metric":
                    "Predicted 90-Day Revenue",

                "Value":
                    customer.get(
                        "Predicted 90-Day Revenue",
                        "Not Available"
                    )
            },

            {
                "Category":
                    "CLV",

                "Metric":
                    "CLV Value Band",

                "Value":
                    customer.get(
                        "CLV Value Band",
                        "Not Available"
                    )
            },

            {
                "Category":
                    "Business Priority",

                "Metric":
                    "Customer Priority",

                "Value":
                    priority[
                        "priority"
                    ]
            },

            {
                "Category":
                    "Business Priority",

                "Metric":
                    "Retention Priority Score",

                "Value":
                    (
                        retention_score
                        if retention_score
                        is not None
                        else
                        "Not Available"
                    )
            },

            {
                "Category":
                    "Business Priority",

                "Metric":
                    "Retention Rank",

                "Value":
                    (
                        retention_rank
                        if retention_rank
                        is not None
                        else
                        "Not Available"
                    )
            },

            {
                "Category":
                    "Recommendation",

                "Metric":
                    "Top Recommended Product",

                "Value":
                    customer.get(
                        "Top Recommended Product",
                        "Not Available"
                    )
            },

            {
                "Category":
                    "Recommendation",

                "Metric":
                    "Recommendation Source",

                "Value":
                    customer.get(
                        "Recommendation Source",
                        "Not Available"
                    )
            },

            {
                "Category":
                    "Business Action",

                "Metric":
                    "Recommended Business Action",

                "Value":
                    priority[
                        "business_action"
                    ]
            }
        ]

        # -----------------------------------------------------
        # Behavioral fields
        # -----------------------------------------------------

        for feature in (
            self.BEHAVIOR_FEATURES
        ):

            if feature in customer.index:

                report_rows.append(
                    {
                        "Category":
                            "Customer Behaviour",

                        "Metric":
                            feature,

                        "Value":
                            customer.get(
                                feature
                            )
                    }
                )

        return pd.DataFrame(
            report_rows
        )

    # =========================================================
    # EXPORT CSV
    # =========================================================

    def customer_report_csv(
        self,
        df: pd.DataFrame,
        customer_id
    ) -> bytes:

        report = (
            self.build_customer_report(
                df,
                customer_id
            )
        )

        if report.empty:

            return b""

        buffer = io.StringIO()

        report.to_csv(
            buffer,
            index=False
        )

        return (
            buffer
            .getvalue()
            .encode(
                "utf-8"
            )
        )

    # =========================================================
    # RENDER
    # =========================================================

    def render(
        self,
        df: pd.DataFrame
    ):

        data = (
            self.prepare_data(
                df
            )
        )

        st.title(
            "360° Customer Explorer"
        )

        st.caption(
            "Unified customer-level intelligence combining "
            "behaviour, segmentation, churn, predicted "
            "90-day revenue and product recommendations."
        )

        # =====================================================
        # CUSTOMER SELECTOR
        # =====================================================

        customer_options = (
            data[
                CUSTOMER_ID
            ]
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        selected_customer = (
            st.selectbox(
                "Select Customer ID",
                options=
                    customer_options,
                key=
                    "final_customer_explorer"
            )
        )

        customer = (
            self.get_customer(
                data,
                selected_customer
            )
        )

        if customer is None:

            st.warning(
                "Customer not found."
            )

            return

        priority = (
            self.customer_priority(
                data,
                selected_customer
            )
        )

        retention_score = (
            self.retention_priority_score(
                data,
                selected_customer
            )
        )

        retention_rank = (
            self.retention_rank(
                data,
                selected_customer
            )
        )

        # =====================================================
        # CUSTOMER IDENTITY
        # =====================================================

        st.subheader(
            "Customer Intelligence Summary"
        )

        col1, col2, col3, col4 = (
            st.columns(
                4
            )
        )

        with col1:

            st.metric(
                "Customer ID",
                customer.get(
                    CUSTOMER_ID,
                    ""
                )
            )

        with col2:

            st.metric(
                "Customer Segment",
                customer.get(
                    "Customer Segment",
                    "N/A"
                )
            )

        with col3:

            st.metric(
                "Customer Priority",
                priority[
                    "priority"
                ]
            )

        with col4:

            st.metric(
                "Retention Rank",
                (
                    f"#{retention_rank}"
                    if retention_rank
                    is not None
                    else
                    "N/A"
                )
            )

        st.divider()

        # =====================================================
        # CHURN + CLV
        # =====================================================

        st.subheader(
            "Risk and Customer Value"
        )

        col1, col2, col3, col4 = (
            st.columns(
                4
            )
        )

        with col1:

            churn_probability = (
                customer.get(
                    "Churn Probability"
                )
            )

            if pd.notna(
                churn_probability
            ):

                churn_text = (
                    f"{float(churn_probability) * 100:.2f}%"
                )

            else:

                churn_text = "N/A"

            st.metric(
                "Churn Probability",
                churn_text
            )

        with col2:

            st.metric(
                "Churn Risk",
                customer.get(
                    "Churn Risk",
                    "N/A"
                )
            )

        with col3:

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

                revenue_text = "N/A"

            st.metric(
                "Predicted 90-Day Revenue",
                revenue_text
            )

        with col4:

            st.metric(
                "CLV Value Band",
                customer.get(
                    "CLV Value Band",
                    "N/A"
                )
            )

        # =====================================================
        # RETENTION PRIORITY
        # =====================================================

        col5, col6 = (
            st.columns(
                2
            )
        )

        with col5:

            st.metric(
                "Retention Priority Score",
                (
                    f"{retention_score:.4f}"
                    if retention_score
                    is not None
                    else
                    "N/A"
                )
            )

        with col6:

            st.write(
                "**Recommended Business Action**"
            )

            st.write(
                priority[
                    "business_action"
                ]
            )

        st.caption(
            "Customer Priority and Retention Priority Score "
            "are decision-support rules built from existing "
            "model outputs. They are not additional ML models."
        )

        st.divider()

        # =====================================================
        # CUSTOMER BEHAVIOUR
        # =====================================================

        st.subheader(
            "Customer Behaviour Profile"
        )

        behavior_columns = (
            st.columns(
                6
            )
        )

        for container, feature in zip(
            behavior_columns,
            self.BEHAVIOR_FEATURES
        ):

            value = (
                customer.get(
                    feature
                )
            )

            with container:

                if pd.isna(
                    value
                ):

                    display_value = "N/A"

                elif isinstance(
                    value,
                    (
                        float,
                        np.floating
                    )
                ):

                    display_value = (
                        f"{float(value):,.2f}"
                    )

                else:

                    display_value = str(
                        value
                    )

                st.metric(
                    feature,
                    display_value
                )

        # =====================================================
        # BEHAVIORAL COMPARISON
        # =====================================================

        percentiles = (
            self.behavior_percentiles(
                data,
                selected_customer
            )
        )

        if not percentiles.empty:

            st.subheader(
                "Customer vs Population"
            )

            comparison_chart = (
                px.bar(
                    percentiles,
                    x=
                        "Feature",
                    y=
                        "Percentile",
                    text=
                        "Percentile",
                    hover_data=[
                        "Customer Value",
                        "Population Average"
                    ],
                    title=
                        "Customer Behaviour Percentile"
                )
            )

            comparison_chart.update_layout(
                yaxis_title=
                    "Population Percentile",

                yaxis_range=[
                    0,
                    100
                ]
            )

            st.plotly_chart(
                comparison_chart,
                use_container_width=True
            )

            st.caption(
                "Percentiles describe where this customer's "
                "feature value sits within the current "
                "customer population. For Recency, a higher "
                "raw percentile means a longer time since "
                "the last purchase."
            )

        st.divider()

        # =====================================================
        # RECOMMENDATION
        # =====================================================

        st.subheader(
            "Personalized Product Recommendations"
        )

        col1, col2, col3 = (
            st.columns(
                3
            )
        )

        with col1:

            st.metric(
                "Top Product",
                customer.get(
                    "Top Recommended Product",
                    "N/A"
                )
            )

        with col2:

            st.metric(
                "Top Stock Code",
                customer.get(
                    "Top Recommended Stock Code",
                    "N/A"
                )
            )

        with col3:

            recommendation_score = (
                customer.get(
                    "Top Recommendation Score"
                )
            )

            if pd.notna(
                recommendation_score
            ):

                score_text = (
                    f"{float(recommendation_score):.4f}"
                )

            else:

                score_text = "N/A"

            st.metric(
                "Top Recommendation Score",
                score_text
            )

        source = (
            customer.get(
                "Recommendation Source"
            )
        )

        if pd.notna(
            source
        ):

            st.write(
                f"**Recommendation Source:** {source}"
            )

        recommendations = (
            self.recommendations(
                data,
                selected_customer
            )
        )

        if recommendations.empty:

            st.info(
                "No Top-N recommendations "
                "available for this customer."
            )

        else:

            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "Recommendation scores are ranking signals, "
            "not purchase probabilities."
        )

        st.divider()

        # =====================================================
        # COMPLETE RECORD
        # =====================================================

        st.subheader(
            "Complete Customer Intelligence Record"
        )

        record = (
            customer
            .to_frame(
                name="Value"
            )
        )

        st.dataframe(
            record,
            use_container_width=True
        )

        st.divider()

        # =====================================================
        # DOWNLOADABLE REPORT
        # =====================================================

        st.subheader(
            "Export Customer Intelligence"
        )

        report = (
            self.build_customer_report(
                data,
                selected_customer
            )
        )

        st.dataframe(
            report,
            use_container_width=True,
            hide_index=True
        )

        report_bytes = (
            self.customer_report_csv(
                data,
                selected_customer
            )
        )

        safe_customer_id = (
            str(
                selected_customer
            )
            .replace(
                " ",
                "_"
            )
            .replace(
                "/",
                "_"
            )
        )

        st.download_button(
            label=
                "Download Customer Intelligence Report",

            data=
                report_bytes,

            file_name=
                (
                    f"customer_"
                    f"{safe_customer_id}_"
                    f"intelligence_report.csv"
                ),

            mime=
                "text/csv"
        )