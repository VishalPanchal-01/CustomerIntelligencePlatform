import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


CUSTOMER_ID = "Customer ID"


class SegmentationIntelligence:

    def prepare_data(self,df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        if CUSTOMER_ID not in data.columns:
            raise ValueError("Customer ID column not found.")
        if ("Customer Segment" not in data.columns):
            raise ValueError("Customer Segment column not found.")

        numeric_columns = [
            "Recency",
            "Frequency",
            "Monetary",
            "TotalItems",
            "AverageOrderValue",
            "Tenure",
            "Churn Probability",
            "Predicted 90-Day Revenue"
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

        data[
            "Customer Segment"
        ] = (
            data[
                "Customer Segment"
            ]
            .fillna(
                "Unknown"
            )
            .astype(str)
        )

        return data

    # =========================================================
    # OVERVIEW METRICS
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

        segment_count = int(
            data[
                "Customer Segment"
            ]
            .nunique()
        )

        # -----------------------------------------------------
        # Largest segment
        # -----------------------------------------------------

        segment_counts = (
            data[
                "Customer Segment"
            ]
            .value_counts()
        )

        if not segment_counts.empty:

            largest_segment = str(
                segment_counts.index[0]
            )

            largest_segment_customers = int(
                segment_counts.iloc[0]
            )

        else:

            largest_segment = "N/A"
            largest_segment_customers = 0

        # -----------------------------------------------------
        # Highest revenue segment
        # -----------------------------------------------------

        if (
            "Predicted 90-Day Revenue"
            in data.columns
        ):

            revenue_by_segment = (
                data
                .groupby(
                    "Customer Segment"
                )[
                    "Predicted 90-Day Revenue"
                ]
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            if not revenue_by_segment.empty:

                highest_revenue_segment = str(
                    revenue_by_segment.index[0]
                )

                highest_revenue_value = float(
                    revenue_by_segment.iloc[0]
                )

            else:

                highest_revenue_segment = "N/A"
                highest_revenue_value = 0.0

        else:

            highest_revenue_segment = "N/A"
            highest_revenue_value = 0.0

        # -----------------------------------------------------
        # Highest churn segment
        # -----------------------------------------------------

        if (
            "Churn Probability"
            in data.columns
        ):

            churn_by_segment = (
                data
                .groupby(
                    "Customer Segment"
                )[
                    "Churn Probability"
                ]
                .mean()
                .sort_values(
                    ascending=False
                )
            )

            if not churn_by_segment.empty:

                highest_churn_segment = str(
                    churn_by_segment.index[0]
                )

                highest_churn_probability = float(
                    churn_by_segment.iloc[0]
                )

            else:

                highest_churn_segment = "N/A"
                highest_churn_probability = 0.0

        else:

            highest_churn_segment = "N/A"
            highest_churn_probability = 0.0

        return {

            "total_customers":
                total_customers,

            "segment_count":
                segment_count,

            "largest_segment":
                largest_segment,

            "largest_segment_customers":
                largest_segment_customers,

            "highest_revenue_segment":
                highest_revenue_segment,

            "highest_revenue_value":
                highest_revenue_value,

            "highest_churn_segment":
                highest_churn_segment,

            "highest_churn_probability":
                highest_churn_probability
        }

    # =========================================================
    # SEGMENT DISTRIBUTION
    # =========================================================

    def segment_distribution(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = (
            self.prepare_data(
                df
            )
        )

        total_customers = max(
            data[
                CUSTOMER_ID
            ]
            .nunique(),
            1
        )

        result = (
            data[
                "Customer Segment"
            ]
            .value_counts()
            .rename_axis(
                "Customer Segment"
            )
            .reset_index(
                name="Customers"
            )
        )

        result[
            "Customer Percentage"
        ] = (
            result[
                "Customers"
            ]
            /
            total_customers
            *
            100
        ).round(
            2
        )

        return result

    # =========================================================
    # SEGMENT PROFILE
    # =========================================================

    def segment_profile(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = (
            self.prepare_data(
                df
            )
        )

        features = [
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

        if not features:

            return pd.DataFrame()

        result = (
            data
            .groupby(
                "Customer Segment"
            )[
                features
            ]
            .mean()
            .round(
                2
            )
            .reset_index()
        )

        return result

    # =========================================================
    # SEGMENT BUSINESS PERFORMANCE
    # =========================================================

    def segment_business_performance(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = (
            self.prepare_data(
                df
            )
        )

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

        data[
            "High Value Indicator"
        ] = (
            data[
                "CLV Value Band"
            ]
            .astype(str)
            .str.lower()
            .eq(
                "high"
            )
            .astype(int)
            if "CLV Value Band" in data.columns
            else 0
        )

        aggregation = {

            CUSTOMER_ID:
                "nunique",

            "High Risk Indicator":
                "sum",

            "High Value Indicator":
                "sum"
        }

        if (
            "Churn Probability"
            in data.columns
        ):

            aggregation[
                "Churn Probability"
            ] = "mean"

        if (
            "Predicted 90-Day Revenue"
            in data.columns
        ):

            aggregation[
                "Predicted 90-Day Revenue"
            ] = [
                "mean",
                "sum"
            ]

        result = (
            data
            .groupby(
                "Customer Segment"
            )
            .agg(
                aggregation
            )
        )

        # -----------------------------------------------------
        # Flatten columns
        # -----------------------------------------------------

        result.columns = [
            (
                " ".join(
                    [
                        str(part)
                        for part in column
                        if str(part) != ""
                    ]
                )
                if isinstance(
                    column,
                    tuple
                )
                else str(
                    column
                )
            )
            for column
            in result.columns
        ]

        result = (
            result
            .reset_index()
        )

        result = result.rename(
            columns={

                f"{CUSTOMER_ID} nunique":
                    "Customers",

                "High Risk Indicator sum":
                    "High Risk Customers",

                "High Value Indicator sum":
                    "High Value Customers",

                "Churn Probability mean":
                    "Average Churn Probability",

                "Predicted 90-Day Revenue mean":
                    "Average Predicted Revenue",

                "Predicted 90-Day Revenue sum":
                    "Total Predicted Revenue"
            }
        )

        # -----------------------------------------------------
        # Percentage metrics
        # -----------------------------------------------------

        if (
            "Customers"
            in result.columns
        ):

            denominator = (
                result[
                    "Customers"
                ]
                .replace(
                    0,
                    np.nan
                )
            )

            result[
                "High Risk %"
            ] = (
                result[
                    "High Risk Customers"
                ]
                /
                denominator
                *
                100
            ).fillna(
                0
            )

            result[
                "High Value %"
            ] = (
                result[
                    "High Value Customers"
                ]
                /
                denominator
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
        ] = (
            result[
                numeric_columns
            ]
            .round(
                2
            )
        )

        return result

    # =========================================================
    # REVENUE CONTRIBUTION
    # =========================================================

    def revenue_contribution(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = (
            self.prepare_data(
                df
            )
        )

        if (
            "Predicted 90-Day Revenue"
            not in data.columns
        ):

            return pd.DataFrame()

        result = (
            data
            .groupby(
                "Customer Segment"
            )
            .agg(
                **{
                    "Customers":
                        (
                            CUSTOMER_ID,
                            "nunique"
                        ),

                    "Total Predicted Revenue":
                        (
                            "Predicted 90-Day Revenue",
                            "sum"
                        ),

                    "Average Predicted Revenue":
                        (
                            "Predicted 90-Day Revenue",
                            "mean"
                        )
                }
            )
            .reset_index()
        )

        total_revenue = float(
            result[
                "Total Predicted Revenue"
            ]
            .sum()
        )

        if total_revenue > 0:

            result[
                "Revenue Contribution %"
            ] = (
                result[
                    "Total Predicted Revenue"
                ]
                /
                total_revenue
                *
                100
            )

        else:

            result[
                "Revenue Contribution %"
            ] = 0.0

        numeric_columns = [
            "Total Predicted Revenue",
            "Average Predicted Revenue",
            "Revenue Contribution %"
        ]

        result[
            numeric_columns
        ] = (
            result[
                numeric_columns
            ]
            .round(
                2
            )
        )

        return (
            result
            .sort_values(
                by=
                    "Total Predicted Revenue",
                ascending=
                    False
            )
            .reset_index(
                drop=True
            )
        )

    # =========================================================
    # CHURN EXPOSURE BY SEGMENT
    # =========================================================

    def churn_exposure(
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

            return pd.DataFrame()

        result = (
            data
            .groupby(
                [
                    "Customer Segment",
                    "Churn Risk"
                ]
            )
            .size()
            .reset_index(
                name="Customers"
            )
        )

        return result

    # =========================================================
    # CLV COMPOSITION
    # =========================================================

    def clv_composition(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = (
            self.prepare_data(
                df
            )
        )

        if (
            "CLV Value Band"
            not in data.columns
        ):

            return pd.DataFrame()

        result = (
            data
            .groupby(
                [
                    "Customer Segment",
                    "CLV Value Band"
                ]
            )
            .size()
            .reset_index(
                name="Customers"
            )
        )

        return result

    # =========================================================
    # SEGMENT PRIORITY PROFILE
    # =========================================================

    def priority_profile(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = (
            self.prepare_data(
                df
            )
        )

        # -----------------------------------------------------
        # Create rule-based priority
        # -----------------------------------------------------

        def assign_priority(
            row
        ):

            churn = str(
                row.get(
                    "Churn Risk",
                    ""
                )
            ).lower()

            clv = str(
                row.get(
                    "CLV Value Band",
                    ""
                )
            ).lower()

            if (
                churn == "high"
                and
                clv == "high"
            ):

                return "Critical"

            if (
                (
                    churn == "high"
                    and
                    clv == "medium"
                )
                or
                (
                    churn == "medium"
                    and
                    clv == "high"
                )
            ):

                return "High"

            if (
                churn == "high"
                or
                clv == "high"
            ):

                return "Medium"

            return "Low"

        data[
            "Customer Priority"
        ] = data.apply(
            assign_priority,
            axis=1
        )

        result = (
            data
            .groupby(
                [
                    "Customer Segment",
                    "Customer Priority"
                ]
            )
            .size()
            .reset_index(
                name="Customers"
            )
        )

        return result

    # =========================================================
    # SEGMENT BUSINESS ACTIONS
    # =========================================================

    def segment_actions(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        performance = (
            self.segment_business_performance(
                df
            )
        )

        if performance.empty:

            return pd.DataFrame()

        rows = []

        for _, row in (
            performance
            .iterrows()
        ):

            segment = (
                row[
                    "Customer Segment"
                ]
            )

            high_risk_percentage = float(
                row.get(
                    "High Risk %",
                    0
                )
            )

            high_value_percentage = float(
                row.get(
                    "High Value %",
                    0
                )
            )

            # -------------------------------------------------
            # Business interpretation
            # -------------------------------------------------

            if (
                high_risk_percentage >= 40
                and
                high_value_percentage >= 30
            ):

                action = (
                    "High-priority retention segment. "
                    "Use personalized retention offers "
                    "and proactive engagement."
                )

            elif (
                high_risk_percentage >= 40
            ):

                action = (
                    "Retention-focused segment. "
                    "Investigate disengagement and "
                    "run reactivation campaigns."
                )

            elif (
                high_value_percentage >= 40
            ):

                action = (
                    "High-value growth segment. "
                    "Use loyalty, cross-sell and "
                    "premium recommendations."
                )

            elif (
                high_value_percentage >= 20
            ):

                action = (
                    "Growth opportunity segment. "
                    "Use targeted cross-sell and "
                    "personalized recommendations."
                )

            else:

                action = (
                    "Maintain engagement and monitor "
                    "behaviour for changes in value "
                    "or churn risk."
                )

            rows.append(
                {
                    "Customer Segment":
                        segment,

                    "High Risk %":
                        high_risk_percentage,

                    "High Value %":
                        high_value_percentage,

                    "Recommended Segment Strategy":
                        action
                }
            )

        return pd.DataFrame(
            rows
        )

    # =========================================================
    # CUSTOMER TABLE BY SEGMENT
    # =========================================================

    def segment_customers(
        self,
        df: pd.DataFrame,
        segment: str,
        top_n: int = 100
    ) -> pd.DataFrame:

        data = (
            self.prepare_data(
                df
            )
        )

        result = (
            data[
                data[
                    "Customer Segment"
                ]
                ==
                segment
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
                    by=
                        "Predicted 90-Day Revenue",

                    ascending=
                        False
                )
            )

        desired_columns = [
            CUSTOMER_ID,
            "Customer Segment",
            "Recency",
            "Frequency",
            "Monetary",
            "AverageOrderValue",
            "Tenure",
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
            "Customer Segmentation Intelligence"
        )

        st.caption(
            "Analyse customer behavioural groups, "
            "segment value, churn exposure and "
            "segment-level business opportunities."
        )

        # =====================================================
        # KPIs
        # =====================================================

        metrics = (
            self.calculate_metrics(
                data
            )
        )

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
                "Customer Segments",
                f"{metrics['segment_count']}"
            )

        with col3:

            st.metric(
                "Largest Segment",
                metrics[
                    "largest_segment"
                ],
                (
                    f"{metrics['largest_segment_customers']} customers"
                )
            )

        with col4:

            st.metric(
                "Highest Revenue Segment",
                metrics[
                    "highest_revenue_segment"
                ],
                (
                    f"{metrics['highest_revenue_value']:,.2f}"
                )
            )

        st.divider()

        # =====================================================
        # SEGMENT DISTRIBUTION
        # =====================================================

        left, right = (
            st.columns(
                2
            )
        )

        distribution = (
            self.segment_distribution(
                data
            )
        )

        with left:

            st.subheader(
                "Segment Distribution"
            )

            distribution_chart = (
                px.bar(
                    distribution,
                    x=
                        "Customer Segment",
                    y=
                        "Customers",
                    text=
                        "Customers",
                    title=
                        "Customers by Segment"
                )
            )

            st.plotly_chart(
                distribution_chart,
                use_container_width=True
            )

        with right:

            st.subheader(
                "Segment Share"
            )

            pie_chart = (
                px.pie(
                    distribution,
                    names=
                        "Customer Segment",
                    values=
                        "Customers",
                    title=
                        "Customer Distribution Across Segments"
                )
            )

            st.plotly_chart(
                pie_chart,
                use_container_width=True
            )

        st.divider()

        # =====================================================
        # BEHAVIOURAL PROFILE
        # =====================================================

        st.subheader(
            "Segment Behavioural Profile"
        )

        profile = (
            self.segment_profile(
                data
            )
        )

        if profile.empty:

            st.info(
                "Behavioural features are not available."
            )

        else:

            st.dataframe(
                profile,
                use_container_width=True,
                hide_index=True
            )

            available_features = [
                column
                for column in [
                    "Recency",
                    "Frequency",
                    "Monetary",
                    "TotalItems",
                    "AverageOrderValue",
                    "Tenure"
                ]
                if column in profile.columns
            ]

            if available_features:

                selected_feature = (
                    st.selectbox(
                        "Select behavioural metric",
                        options=
                            available_features,
                        key=
                            "segment_behavior_metric"
                    )
                )

                feature_chart = (
                    px.bar(
                        profile,
                        x=
                            "Customer Segment",
                        y=
                            selected_feature,
                        text=
                            selected_feature,
                        title=
                            (
                                f"Average {selected_feature} "
                                f"by Segment"
                            )
                    )
                )

                st.plotly_chart(
                    feature_chart,
                    use_container_width=True
                )

        st.divider()

        # =====================================================
        # REVENUE CONTRIBUTION
        # =====================================================

        st.subheader(
            "Segment Revenue Contribution"
        )

        revenue = (
            self.revenue_contribution(
                data
            )
        )

        if revenue.empty:

            st.info(
                "Predicted revenue data is not available."
            )

        else:

            col1, col2 = (
                st.columns(
                    2
                )
            )

            with col1:

                revenue_chart = (
                    px.bar(
                        revenue,
                        x=
                            "Customer Segment",
                        y=
                            "Total Predicted Revenue",
                        text=
                            "Total Predicted Revenue",
                        title=
                            "Predicted 90-Day Revenue by Segment"
                    )
                )

                st.plotly_chart(
                    revenue_chart,
                    use_container_width=True
                )

            with col2:

                revenue_share = (
                    px.pie(
                        revenue,
                        names=
                            "Customer Segment",
                        values=
                            "Total Predicted Revenue",
                        title=
                            "Revenue Contribution Share"
                    )
                )

                st.plotly_chart(
                    revenue_share,
                    use_container_width=True
                )

            st.dataframe(
                revenue,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        # =====================================================
        # CHURN EXPOSURE
        # =====================================================

        st.subheader(
            "Churn Exposure by Segment"
        )

        churn_data = (
            self.churn_exposure(
                data
            )
        )

        if churn_data.empty:

            st.info(
                "Churn Risk data is not available."
            )

        else:

            churn_chart = (
                px.bar(
                    churn_data,
                    x=
                        "Customer Segment",
                    y=
                        "Customers",
                    color=
                        "Churn Risk",
                    barmode=
                        "group",
                    title=
                        "Churn Risk Composition by Segment"
                )
            )

            st.plotly_chart(
                churn_chart,
                use_container_width=True
            )

        st.divider()

        # =====================================================
        # CLV COMPOSITION
        # =====================================================

        st.subheader(
            "CLV Composition by Segment"
        )

        clv_data = (
            self.clv_composition(
                data
            )
        )

        if clv_data.empty:

            st.info(
                "CLV Value Band data is not available."
            )

        else:

            clv_chart = (
                px.bar(
                    clv_data,
                    x=
                        "Customer Segment",
                    y=
                        "Customers",
                    color=
                        "CLV Value Band",
                    barmode=
                        "group",
                    title=
                        "Customer Value Composition by Segment"
                )
            )

            st.plotly_chart(
                clv_chart,
                use_container_width=True
            )

        st.divider()

        # =====================================================
        # BUSINESS PERFORMANCE
        # =====================================================

        st.subheader(
            "Segment Business Performance"
        )

        performance = (
            self.segment_business_performance(
                data
            )
        )

        st.dataframe(
            performance,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # =====================================================
        # SEGMENT STRATEGIES
        # =====================================================

        st.subheader(
            "Recommended Segment Strategies"
        )

        strategies = (
            self.segment_actions(
                data
            )
        )

        if strategies.empty:

            st.info(
                "Segment strategy data is unavailable."
            )

        else:

            st.dataframe(
                strategies,
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "Recommended segment strategies are "
            "rule-based decision-support suggestions "
            "derived from the segment's current churn "
            "and value profile. They are not ML predictions."
        )

        st.divider()

        # =====================================================
        # SEGMENT EXPLORER
        # =====================================================

        st.subheader(
            "Segment Customer Explorer"
        )

        segment_options = (
            sorted(
                data[
                    "Customer Segment"
                ]
                .dropna()
                .unique()
                .tolist()
            )
        )

        selected_segment = (
            st.selectbox(
                "Select Customer Segment",
                options=
                    segment_options,
                key=
                    "segment_customer_explorer"
            )
        )

        segment_customer_table = (
            self.segment_customers(
                data,
                segment=
                    selected_segment,
                top_n=
                    100
            )
        )

        st.dataframe(
            segment_customer_table,
            use_container_width=True,
            hide_index=True
        )