import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


CUSTOMER_ID = "Customer ID"


class ChurnIntelligence:

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

        # -----------------------------------------------------
        # Churn Probability
        # -----------------------------------------------------

        if (
            "Churn Probability"
            in data.columns
        ):

            data[
                "Churn Probability"
            ] = pd.to_numeric(
                data[
                    "Churn Probability"
                ],
                errors="coerce"
            )

        # -----------------------------------------------------
        # Predicted Revenue
        # -----------------------------------------------------

        if (
            "Predicted 90-Day Revenue"
            in data.columns
        ):

            data[
                "Predicted 90-Day Revenue"
            ] = pd.to_numeric(
                data[
                    "Predicted 90-Day Revenue"
                ],
                errors="coerce"
            )

        # -----------------------------------------------------
        # Customer behavioural variables
        # -----------------------------------------------------

        numeric_columns = [
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
    # CHURN METRICS
    # =========================================================

    def calculate_metrics(
        self,
        df: pd.DataFrame
    ) -> dict:

        data = (
            self.prepare_data(
                df
            )
        )

        total_customers = int(
            data[
                CUSTOMER_ID
            ]
            .nunique()
        )

        # -----------------------------------------------------
        # Average churn probability
        # -----------------------------------------------------

        if (
            "Churn Probability"
            in data.columns
        ):

            average_churn_probability = float(
                data[
                    "Churn Probability"
                ]
                .mean()
            )

        else:

            average_churn_probability = 0.0

        # -----------------------------------------------------
        # Risk counts
        # -----------------------------------------------------

        if (
            "Churn Risk"
            in data.columns
        ):

            normalized_risk = (
                data[
                    "Churn Risk"
                ]
                .fillna("")
                .astype(str)
                .str.lower()
            )

            high_risk_customers = int(
                (
                    normalized_risk
                    ==
                    "high"
                )
                .sum()
            )

            medium_risk_customers = int(
                (
                    normalized_risk
                    ==
                    "medium"
                )
                .sum()
            )

            low_risk_customers = int(
                (
                    normalized_risk
                    ==
                    "low"
                )
                .sum()
            )

        else:

            high_risk_customers = 0
            medium_risk_customers = 0
            low_risk_customers = 0

        # -----------------------------------------------------
        # High-risk percentage
        # -----------------------------------------------------

        if total_customers > 0:

            high_risk_percentage = (
                high_risk_customers
                /
                total_customers
                *
                100
            )

        else:

            high_risk_percentage = 0.0

        # -----------------------------------------------------
        # Revenue at risk
        # -----------------------------------------------------

        if (
            "Predicted 90-Day Revenue"
            in data.columns
            and
            "Churn Risk"
            in data.columns
        ):

            high_risk_mask = (
                data[
                    "Churn Risk"
                ]
                .astype(str)
                .str.lower()
                .eq(
                    "high"
                )
            )

            revenue_at_risk = float(
                data.loc[
                    high_risk_mask,
                    "Predicted 90-Day Revenue"
                ]
                .fillna(
                    0
                )
                .sum()
            )

        else:

            revenue_at_risk = 0.0

        # -----------------------------------------------------
        # High-value customers at risk
        # -----------------------------------------------------

        if (
            "CLV Value Band"
            in data.columns
            and
            "Churn Risk"
            in data.columns
        ):

            high_value_at_risk = int(
                (
                    data[
                        "CLV Value Band"
                    ]
                    .astype(str)
                    .str.lower()
                    .eq(
                        "high"
                    )
                    &
                    data[
                        "Churn Risk"
                    ]
                    .astype(str)
                    .str.lower()
                    .eq(
                        "high"
                    )
                )
                .sum()
            )

        else:

            high_value_at_risk = 0

        return {

            "total_customers":
                total_customers,

            "average_churn_probability":
                average_churn_probability,

            "high_risk_customers":
                high_risk_customers,

            "medium_risk_customers":
                medium_risk_customers,

            "low_risk_customers":
                low_risk_customers,

            "high_risk_percentage":
                float(
                    high_risk_percentage
                ),

            "revenue_at_risk":
                revenue_at_risk,

            "high_value_at_risk":
                high_value_at_risk
        }

    # =========================================================
    # RISK DISTRIBUTION
    # =========================================================

    def risk_distribution(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = (
            self.prepare_data(
                df
            )
        )

        if (
            "Churn Risk"
            not in data.columns
        ):

            return pd.DataFrame(
                columns=[
                    "Churn Risk",
                    "Customers"
                ]
            )

        risk_order = [
            "High",
            "Medium",
            "Low"
        ]

        result = (
            data[
                "Churn Risk"
            ]
            .value_counts()
            .reindex(
                risk_order,
                fill_value=0
            )
            .rename_axis(
                "Churn Risk"
            )
            .reset_index(
                name="Customers"
            )
        )

        return result

    # =========================================================
    # CHURN BY SEGMENT
    # =========================================================

    def churn_by_segment(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = (
            self.prepare_data(
                df
            )
        )

        if (
            "Customer Segment"
            not in data.columns
            or
            "Churn Probability"
            not in data.columns
        ):

            return pd.DataFrame()

        data[
            "High Risk Indicator"
        ] = (
            data[
                "Churn Risk"
            ]
            .astype(str)
            .str.lower()
            .eq(
                "high"
            )
            .astype(int)
            if "Churn Risk" in data.columns
            else 0
        )

        aggregation = {

            CUSTOMER_ID:
                "nunique",

            "Churn Probability":
                "mean",

            "High Risk Indicator":
                "sum"
        }

        if (
            "Predicted 90-Day Revenue"
            in data.columns
        ):

            aggregation[
                "Predicted 90-Day Revenue"
            ] = "sum"

        result = (
            data
            .groupby(
                "Customer Segment"
            )
            .agg(
                aggregation
            )
            .reset_index()
        )

        result = result.rename(
            columns={
                CUSTOMER_ID:
                    "Customers",

                "Churn Probability":
                    "Average Churn Probability",

                "High Risk Indicator":
                    "High Risk Customers",

                "Predicted 90-Day Revenue":
                    "Total Predicted Revenue"
            }
        )

        result[
            "High Risk %"
        ] = (
            result[
                "High Risk Customers"
            ]
            /
            result[
                "Customers"
            ]
            .replace(
                0,
                np.nan
            )
            *
            100
        ).fillna(
            0
        )

        numeric_columns = (
            result
            .select_dtypes(
                include="number"
            )
            .columns
        )

        result[
            numeric_columns
        ] = result[
            numeric_columns
        ].round(
            2
        )

        result = (
            result
            .sort_values(
                by=
                    "Average Churn Probability",

                ascending=
                    False
            )
            .reset_index(
                drop=True
            )
        )

        return result

    # =========================================================
    # REVENUE AT RISK BY SEGMENT
    # =========================================================

    def revenue_at_risk_by_segment(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = (
            self.prepare_data(
                df
            )
        )

        required = [
            "Customer Segment",
            "Churn Risk",
            "Predicted 90-Day Revenue"
        ]

        if any(
            column not in data.columns
            for column in required
        ):

            return pd.DataFrame()

        high_risk = (
            data[
                data[
                    "Churn Risk"
                ]
                .astype(str)
                .str.lower()
                .eq(
                    "high"
                )
            ]
            .copy()
        )

        if high_risk.empty:

            return pd.DataFrame(
                columns=[
                    "Customer Segment",
                    "High Risk Customers",
                    "Revenue at Risk"
                ]
            )

        result = (
            high_risk
            .groupby(
                "Customer Segment"
            )
            .agg(
                **{
                    "High Risk Customers":
                        (
                            CUSTOMER_ID,
                            "nunique"
                        ),

                    "Revenue at Risk":
                        (
                            "Predicted 90-Day Revenue",
                            "sum"
                        )
                }
            )
            .reset_index()
        )

        result[
            "Revenue at Risk"
        ] = (
            result[
                "Revenue at Risk"
            ]
            .round(
                2
            )
        )

        result = (
            result
            .sort_values(
                by=
                    "Revenue at Risk",

                ascending=
                    False
            )
            .reset_index(
                drop=True
            )
        )

        return result

    # =========================================================
    # HIGH VALUE AT RISK
    # =========================================================

    def high_value_customers_at_risk(
        self,
        df: pd.DataFrame,
        top_n: int = 50
    ) -> pd.DataFrame:

        data = (
            self.prepare_data(
                df
            )
        )

        required = [
            "Churn Risk",
            "CLV Value Band"
        ]

        if any(
            column not in data.columns
            for column in required
        ):

            return pd.DataFrame()

        mask = (
            data[
                "Churn Risk"
            ]
            .astype(str)
            .str.lower()
            .eq(
                "high"
            )
            &
            data[
                "CLV Value Band"
            ]
            .astype(str)
            .str.lower()
            .eq(
                "high"
            )
        )

        result = (
            data[
                mask
            ]
            .copy()
        )

        if (
            "Predicted 90-Day Revenue"
            in result.columns
        ):

            result = (
                result
                .sort_values(
                    by=[
                        "Churn Probability",
                        "Predicted 90-Day Revenue"
                    ],
                    ascending=[
                        False,
                        False
                    ]
                )
            )

        elif (
            "Churn Probability"
            in result.columns
        ):

            result = (
                result
                .sort_values(
                    by=
                        "Churn Probability",

                    ascending=
                        False
                )
            )

        desired_columns = [
            CUSTOMER_ID,
            "Customer Segment",
            "Churn Probability",
            "Churn Risk",
            "Predicted 90-Day Revenue",
            "CLV Value Band",
            "Top Recommended Product"
        ]

        available_columns = [
            column
            for column in desired_columns
            if column in result.columns
        ]

        return (
            result[
                available_columns
            ]
            .head(
                top_n
            )
            .reset_index(
                drop=True
            )
        )

    # =========================================================
    # RETENTION PRIORITY TABLE
    # =========================================================

    def retention_priority_customers(
        self,
        df: pd.DataFrame,
        top_n: int = 100
    ) -> pd.DataFrame:

        data = (
            self.prepare_data(
                df
            )
        )

        if (
            "Churn Probability"
            not in data.columns
        ):

            return pd.DataFrame()

        # -----------------------------------------------------
        # Revenue
        # -----------------------------------------------------

        if (
            "Predicted 90-Day Revenue"
            in data.columns
        ):

            revenue = (
                data[
                    "Predicted 90-Day Revenue"
                ]
                .fillna(
                    0
                )
                .clip(
                    lower=0
                )
            )

            maximum_revenue = float(
                revenue.max()
            )

            if maximum_revenue > 0:

                normalized_revenue = (
                    revenue
                    /
                    maximum_revenue
                )

            else:

                normalized_revenue = pd.Series(
                    np.zeros(
                        len(data)
                    ),
                    index=data.index
                )

        else:

            normalized_revenue = pd.Series(
                np.zeros(
                    len(data)
                ),
                index=data.index
            )

        # -----------------------------------------------------
        # Priority score
        #
        # 70% churn risk
        # 30% predicted value
        # -----------------------------------------------------

        churn_probability = (
            data[
                "Churn Probability"
            ]
            .fillna(
                0
            )
            .clip(
                lower=0,
                upper=1
            )
        )

        data[
            "Retention Priority Score"
        ] = (
            0.70
            *
            churn_probability
            +
            0.30
            *
            normalized_revenue
        )

        data[
            "Retention Priority Score"
        ] = (
            data[
                "Retention Priority Score"
            ]
            .round(
                4
            )
        )

        result = (
            data
            .sort_values(
                by=
                    "Retention Priority Score",

                ascending=
                    False
            )
            .head(
                top_n
            )
        )

        desired_columns = [
            CUSTOMER_ID,
            "Customer Segment",
            "Churn Probability",
            "Churn Risk",
            "Predicted 90-Day Revenue",
            "CLV Value Band",
            "Retention Priority Score",
            "Top Recommended Product"
        ]

        available_columns = [
            column
            for column in desired_columns
            if column in result.columns
        ]

        return (
            result[
                available_columns
            ]
            .reset_index(
                drop=True
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
            "Churn Intelligence"
        )

        st.caption(
            "Customer retention intelligence combining "
            "predicted churn probability, customer value, "
            "behaviour and revenue-at-risk analysis."
        )

        if (
            "Churn Probability"
            not in data.columns
        ):

            st.warning(
                "Churn Probability data is not available."
            )

            return

        metrics = (
            self.calculate_metrics(
                data
            )
        )

        # =====================================================
        # KPI ROW 1
        # =====================================================

        col1, col2, col3, col4 = (
            st.columns(
                4
            )
        )

        with col1:

            st.metric(
                "Customers",
                f"{metrics['total_customers']:,}"
            )

        with col2:

            st.metric(
                "Average Churn Probability",
                (
                    f"{metrics['average_churn_probability'] * 100:.2f}%"
                )
            )

        with col3:

            st.metric(
                "High Risk Customers",
                f"{metrics['high_risk_customers']:,}"
            )

        with col4:

            st.metric(
                "High Risk %",
                (
                    f"{metrics['high_risk_percentage']:.2f}%"
                )
            )

        # =====================================================
        # KPI ROW 2
        # =====================================================

        col5, col6, col7 = (
            st.columns(
                3
            )
        )

        with col5:

            st.metric(
                "90-Day Revenue at Risk",
                (
                    f"{metrics['revenue_at_risk']:,.2f}"
                )
            )

        with col6:

            st.metric(
                "High-Value Customers at Risk",
                f"{metrics['high_value_at_risk']:,}"
            )

        with col7:

            st.metric(
                "Medium Risk Customers",
                f"{metrics['medium_risk_customers']:,}"
            )

        st.divider()

        # =====================================================
        # DISTRIBUTION CHARTS
        # =====================================================

        left, right = (
            st.columns(
                2
            )
        )

        # -----------------------------------------------------
        # Churn Probability
        # -----------------------------------------------------

        with left:

            st.subheader(
                "Churn Probability Distribution"
            )

            probability_data = (
                data[
                    data[
                        "Churn Probability"
                    ]
                    .notna()
                ]
            )

            probability_chart = (
                px.histogram(
                    probability_data,
                    x=
                        "Churn Probability",
                    nbins=
                        30,
                    title=
                        "Customer Churn Probability"
                )
            )

            probability_chart.update_layout(
                xaxis_title=
                    "Churn Probability",

                yaxis_title=
                    "Customers"
            )

            st.plotly_chart(
                probability_chart,
                use_container_width=True
            )

        # -----------------------------------------------------
        # Risk Distribution
        # -----------------------------------------------------

        with right:

            st.subheader(
                "Churn Risk Distribution"
            )

            risk_data = (
                self.risk_distribution(
                    data
                )
            )

            risk_chart = (
                px.bar(
                    risk_data,
                    x=
                        "Churn Risk",
                    y=
                        "Customers",
                    text=
                        "Customers",
                    title=
                        "Customers by Churn Risk"
                )
            )

            st.plotly_chart(
                risk_chart,
                use_container_width=True
            )

        st.divider()

        # =====================================================
        # SEGMENT ANALYSIS
        # =====================================================

        st.subheader(
            "Churn by Customer Segment"
        )

        segment_data = (
            self.churn_by_segment(
                data
            )
        )

        if segment_data.empty:

            st.info(
                "Customer Segment data is not available."
            )

        else:

            col1, col2 = (
                st.columns(
                    2
                )
            )

            with col1:

                segment_chart = (
                    px.bar(
                        segment_data,
                        x=
                            "Customer Segment",
                        y=
                            "Average Churn Probability",
                        text=
                            "Average Churn Probability",
                        title=
                            (
                                "Average Churn Probability "
                                "by Segment"
                            )
                    )
                )

                st.plotly_chart(
                    segment_chart,
                    use_container_width=True
                )

            with col2:

                high_risk_chart = (
                    px.bar(
                        segment_data,
                        x=
                            "Customer Segment",
                        y=
                            "High Risk Customers",
                        text=
                            "High Risk Customers",
                        title=
                            "High-Risk Customers by Segment"
                    )
                )

                st.plotly_chart(
                    high_risk_chart,
                    use_container_width=True
                )

            st.dataframe(
                segment_data,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        # =====================================================
        # CUSTOMER BEHAVIOUR VS CHURN
        # =====================================================

        st.subheader(
            "Customer Behaviour vs Churn Probability"
        )

        behavior_options = [
            column
            for column in [
                "Recency",
                "Frequency",
                "Monetary",
                "TotalItems",
                "AverageOrderValue",
                "Tenure"
            ]
            if column in data.columns
        ]

        if behavior_options:

            selected_feature = (
                st.selectbox(
                    "Select behavioural feature",
                    options=
                        behavior_options,
                    key=
                        "churn_behavior_feature"
                )
            )

            plot_data = (
                data[
                    data[
                        selected_feature
                    ]
                    .notna()
                    &
                    data[
                        "Churn Probability"
                    ]
                    .notna()
                ]
                .copy()
            )

            hover_data = [
                CUSTOMER_ID
            ]

            for column in [
                "Customer Segment",
                "Churn Risk",
                "CLV Value Band",
                "Predicted 90-Day Revenue"
            ]:

                if column in plot_data.columns:

                    hover_data.append(
                        column
                    )

            scatter = (
                px.scatter(
                    plot_data,
                    x=
                        selected_feature,
                    y=
                        "Churn Probability",
                    color=(
                        "Churn Risk"
                        if "Churn Risk" in plot_data.columns
                        else None
                    ),
                    hover_data=
                        hover_data,
                    title=
                        (
                            f"{selected_feature} "
                            f"vs Churn Probability"
                        )
                )
            )

            st.plotly_chart(
                scatter,
                use_container_width=True
            )

        else:

            st.info(
                "Customer behavioural features "
                "are not available."
            )

        st.divider()

        # =====================================================
        # REVENUE AT RISK
        # =====================================================

        st.subheader(
            "Revenue at Risk by Customer Segment"
        )

        risk_revenue = (
            self.revenue_at_risk_by_segment(
                data
            )
        )

        if risk_revenue.empty:

            st.info(
                "Revenue-at-risk analysis requires "
                "segment, churn risk and predicted revenue."
            )

        else:

            revenue_chart = (
                px.bar(
                    risk_revenue,
                    x=
                        "Customer Segment",
                    y=
                        "Revenue at Risk",
                    text=
                        "Revenue at Risk",
                    hover_data=[
                        "High Risk Customers"
                    ],
                    title=
                        "Predicted 90-Day Revenue at Risk"
                )
            )

            st.plotly_chart(
                revenue_chart,
                use_container_width=True
            )

            st.dataframe(
                risk_revenue,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        # =====================================================
        # HIGH-VALUE AT-RISK CUSTOMERS
        # =====================================================

        st.subheader(
            "High-Value Customers at Risk"
        )

        high_value_risk = (
            self.high_value_customers_at_risk(
                data,
                top_n=50
            )
        )

        if high_value_risk.empty:

            st.info(
                "No High CLV + High Churn Risk "
                "customers found for the current filters."
            )

        else:

            st.dataframe(
                high_value_risk,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        # =====================================================
        # RETENTION PRIORITY
        # =====================================================

        st.subheader(
            "Retention Priority Customers"
        )

        st.caption(
            "Retention Priority Score combines "
            "70% churn probability and "
            "30% normalized predicted 90-day revenue. "
            "This is a decision-support ranking rule, "
            "not a separate machine-learning model."
        )

        retention_table = (
            self.retention_priority_customers(
                data,
                top_n=100
            )
        )

        if retention_table.empty:

            st.info(
                "Retention priority data is not available."
            )

        else:

            st.dataframe(
                retention_table,
                use_container_width=True,
                hide_index=True
            )