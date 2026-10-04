import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.ui import (
    render_color_card,
    format_risk_badge,
    format_clv_badge
)


CUSTOMER_ID = "Customer ID"


class SegmentationIntelligence:

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

        if "Customer Segment" not in data.columns:
            raise ValueError(
                "Customer Segment column not found."
            )

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

                data[column] = pd.to_numeric(
                    data[column],
                    errors="coerce"
                )

        data["Customer Segment"] = (
            data["Customer Segment"]
            .fillna("Unknown")
            .astype(str)
        )

        return data


    # =========================================================
    # METRICS
    # =========================================================

    def calculate_metrics(
        self,
        df: pd.DataFrame
    ) -> dict:

        data = self.prepare_data(
            df
        )

        total_customers = int(
            data[CUSTOMER_ID].nunique()
        )

        segment_count = int(
            data["Customer Segment"].nunique()
        )

        segment_counts = (
            data["Customer Segment"]
            .value_counts()
        )

        if segment_counts.empty:

            largest_segment = "N/A"
            largest_segment_customers = 0

        else:

            largest_segment = str(
                segment_counts.index[0]
            )

            largest_segment_customers = int(
                segment_counts.iloc[0]
            )

        # -----------------------------------------------------
        # Highest revenue segment
        # -----------------------------------------------------

        if "Predicted 90-Day Revenue" in data.columns:

            revenue_by_segment = (
                data
                .groupby(
                    "Customer Segment"
                )["Predicted 90-Day Revenue"]
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            if revenue_by_segment.empty:

                highest_revenue_segment = "N/A"
                highest_revenue_value = 0.0

            else:

                highest_revenue_segment = str(
                    revenue_by_segment.index[0]
                )

                highest_revenue_value = float(
                    revenue_by_segment.iloc[0]
                )

        else:

            highest_revenue_segment = "N/A"
            highest_revenue_value = 0.0

        # -----------------------------------------------------
        # Highest churn segment
        # -----------------------------------------------------

        if "Churn Probability" in data.columns:

            churn_by_segment = (
                data
                .groupby(
                    "Customer Segment"
                )["Churn Probability"]
                .mean()
                .sort_values(
                    ascending=False
                )
            )

            if churn_by_segment.empty:

                highest_churn_segment = "N/A"
                highest_churn_probability = 0.0

            else:

                highest_churn_segment = str(
                    churn_by_segment.index[0]
                )

                highest_churn_probability = float(
                    churn_by_segment.iloc[0]
                )

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

        data = self.prepare_data(
            df
        )

        total = max(
            data[CUSTOMER_ID].nunique(),
            1
        )

        result = (
            data["Customer Segment"]
            .value_counts()
            .rename_axis(
                "Customer Segment"
            )
            .reset_index(
                name="Customers"
            )
        )

        result["Customer Percentage"] = (
            result["Customers"]
            /
            total
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

        data = self.prepare_data(
            df
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

        return (
            data
            .groupby(
                "Customer Segment"
            )[features]
            .mean()
            .round(
                2
            )
            .reset_index()
        )


    # =========================================================
    # NORMALIZED PROFILE FOR RADAR
    # =========================================================

    def normalized_segment_profile(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        profile = self.segment_profile(
            df
        )

        if profile.empty:
            return pd.DataFrame()

        features = [
            column
            for column in profile.columns
            if column != "Customer Segment"
        ]

        normalized = profile.copy()

        for feature in features:

            minimum = float(
                profile[feature].min()
            )

            maximum = float(
                profile[feature].max()
            )

            if maximum > minimum:

                normalized[feature] = (
                    (
                        profile[feature]
                        -
                        minimum
                    )
                    /
                    (
                        maximum
                        -
                        minimum
                    )
                    *
                    100
                )

            else:

                normalized[feature] = 50.0

        # Recency is inverse:
        # lower recency = more recently active
        if "Recency" in normalized.columns:

            normalized["Recency"] = (
                100
                -
                normalized["Recency"]
            )

        return normalized


    # =========================================================
    # BUSINESS PERFORMANCE
    # =========================================================

    def segment_business_performance(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = self.prepare_data(
            df
        )

        if "Churn Risk" in data.columns:

            data["High Risk Indicator"] = (
                data["Churn Risk"]
                .astype(str)
                .str.lower()
                .eq("high")
                .astype(int)
            )

        else:

            data["High Risk Indicator"] = 0

        if "CLV Value Band" in data.columns:

            data["High Value Indicator"] = (
                data["CLV Value Band"]
                .astype(str)
                .str.lower()
                .eq("high")
                .astype(int)
            )

        else:

            data["High Value Indicator"] = 0

        aggregation = {
            CUSTOMER_ID:
                "nunique",

            "High Risk Indicator":
                "sum",

            "High Value Indicator":
                "sum"
        }

        if "Churn Probability" in data.columns:

            aggregation[
                "Churn Probability"
            ] = "mean"

        if "Predicted 90-Day Revenue" in data.columns:

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
                else str(column)
            )
            for column in result.columns
        ]

        result = result.reset_index()

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

        if "Customers" in result.columns:

            denominator = (
                result["Customers"]
                .replace(
                    0,
                    np.nan
                )
            )

            result["High Risk %"] = (
                result["High Risk Customers"]
                /
                denominator
                *
                100
            ).fillna(
                0
            )

            result["High Value %"] = (
                result["High Value Customers"]
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

        result[numeric_columns] = (
            result[numeric_columns]
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

        data = self.prepare_data(
            df
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

        result[numeric_columns] = (
            result[numeric_columns]
            .round(
                2
            )
        )

        return (
            result
            .sort_values(
                by=
                    "Total Predicted Revenue",

                ascending=False
            )
            .reset_index(
                drop=True
            )
        )


    # =========================================================
    # CHURN COMPOSITION
    # =========================================================

    def churn_exposure(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = self.prepare_data(
            df
        )

        if "Churn Risk" not in data.columns:

            return pd.DataFrame()

        return (
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


    # =========================================================
    # CLV COMPOSITION
    # =========================================================

    def clv_composition(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = self.prepare_data(
            df
        )

        if "CLV Value Band" not in data.columns:

            return pd.DataFrame()

        return (
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


    # =========================================================
    # SEGMENT STRATEGY
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

        for _, row in performance.iterrows():

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

            if (
                high_risk_percentage >= 40
                and
                high_value_percentage >= 30
            ):

                action = (
                    "Immediate retention focus with "
                    "personalized offers and proactive outreach."
                )

                strategy_type = (
                    "Retention Priority"
                )

            elif high_risk_percentage >= 40:

                action = (
                    "Run reactivation campaigns and "
                    "investigate declining customer engagement."
                )

                strategy_type = (
                    "Reactivation"
                )

            elif high_value_percentage >= 40:

                action = (
                    "Use loyalty programs, premium offers "
                    "and personalized cross-sell opportunities."
                )

                strategy_type = (
                    "Growth"
                )

            elif high_value_percentage >= 20:

                action = (
                    "Develop customer value through targeted "
                    "cross-sell and recommendation campaigns."
                )

                strategy_type = (
                    "Development"
                )

            else:

                action = (
                    "Maintain regular engagement and monitor "
                    "changes in customer behaviour."
                )

                strategy_type = (
                    "Maintain"
                )

            rows.append(
                {
                    "Customer Segment":
                        row[
                            "Customer Segment"
                        ],

                    "High Risk %":
                        high_risk_percentage,

                    "High Value %":
                        high_value_percentage,

                    "Strategy":
                        strategy_type,

                    "Recommended Segment Strategy":
                        action
                }
            )

        return pd.DataFrame(
            rows
        )


    # =========================================================
    # SEGMENT CUSTOMERS
    # =========================================================

    def segment_customers(
        self,
        df: pd.DataFrame,
        segment: str,
        top_n: int = 100
    ) -> pd.DataFrame:

        data = self.prepare_data(
            df
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

            result = result.sort_values(
                by=
                    "Predicted 90-Day Revenue",

                ascending=False
            )

        desired_columns = [
            CUSTOMER_ID,
            "Customer Segment",
            "Recency",
            "Frequency",
            "Monetary",
            "TotalItems",
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
    # COLOR KPI CARDS
    # =========================================================

    def _render_kpis(
        self,
        metrics: dict
    ):

        cols = st.columns(
            4
        )

        with cols[0]:

            render_color_card(
                title=
                    "Customers",

                value=
                    f"{metrics['total_customers']:,}",

                icon=
                    "👥",

                card_class=
                    "card-blue"
            )

        with cols[1]:

            render_color_card(
                title=
                    "Customer Segments",

                value=
                    metrics[
                        "segment_count"
                    ],

                icon=
                    "🧩",

                card_class=
                    "card-purple"
            )

        with cols[2]:

            render_color_card(
                title=
                    "Largest Segment",

                value=
                    metrics[
                        "largest_segment"
                    ],

                icon=
                    "🏆",

                card_class=
                    "card-green"
            )

        with cols[3]:

            render_color_card(
                title=
                    "Highest Revenue Segment",

                value=
                    metrics[
                        "highest_revenue_segment"
                    ],

                icon=
                    "💰",

                card_class=
                    "card-green"
            )

        st.write("")

        cols2 = st.columns(
            2
        )

        with cols2[0]:

            render_color_card(
                title=
                    "Customers in Largest Segment",

                value=
                    f"{metrics['largest_segment_customers']:,}",

                icon=
                    "📊",

                card_class=
                    "card-blue"
            )

        with cols2[1]:

            render_color_card(
                title=
                    "Highest Avg Churn Segment",

                value=
                    metrics[
                        "highest_churn_segment"
                    ],

                icon=
                    "⚠️",

                card_class=
                    "card-red"
            )


    # =========================================================
    # RADAR CHART
    # =========================================================

    def _segment_radar(
        self,
        df: pd.DataFrame
    ):

        profile = (
            self.normalized_segment_profile(
                df
            )
        )

        if profile.empty:
            return None

        features = [
            column
            for column in profile.columns
            if column != "Customer Segment"
        ]

        figure = go.Figure()

        for _, row in profile.iterrows():

            values = [
                float(
                    row[feature]
                )
                for feature in features
            ]

            values += [
                values[0]
            ]

            categories = (
                features
                +
                [
                    features[0]
                ]
            )

            figure.add_trace(
                go.Scatterpolar(
                    r=
                        values,

                    theta=
                        categories,

                    fill=
                        "toself",

                    name=
                        row[
                            "Customer Segment"
                        ],

                    opacity=
                        0.55
                )
            )

        figure.update_layout(
            polar={
                "radialaxis": {
                    "visible":
                        True,

                    "range": [
                        0,
                        100
                    ]
                }
            },

            title=
                "Normalized Behavioral Profile",

            height=
                550,

            showlegend=
                True
        )

        return figure


    # =========================================================
    # SEGMENT TREEMAP
    # =========================================================

    def _segment_treemap(
        self,
        df: pd.DataFrame
    ):

        distribution = (
            self.segment_distribution(
                df
            )
        )

        if distribution.empty:
            return None

        figure = px.treemap(
            distribution,

            path=[
                "Customer Segment"
            ],

            values=
                "Customers",

            color=
                "Customers",

            color_continuous_scale=
                "Viridis",

            title=
                "Customer Segment Size"
        )

        figure.update_layout(
            height=450
        )

        return figure


    # =========================================================
    # STYLED CUSTOMER TABLE
    # =========================================================

    def _styled_customer_table(
        self,
        table: pd.DataFrame
    ):

        if table.empty:

            st.info(
                "No customers available "
                "for this segment."
            )

            return

        display = table.copy()

        if "Churn Risk" in display.columns:

            display["Churn Risk"] = (
                display["Churn Risk"]
                .apply(
                    format_risk_badge
                )
            )

        if "CLV Value Band" in display.columns:

            display["CLV Value Band"] = (
                display["CLV Value Band"]
                .apply(
                    format_clv_badge
                )
            )

        if "Churn Probability" in display.columns:

            display["Churn Probability"] = (
                pd.to_numeric(
                    display[
                        "Churn Probability"
                    ],
                    errors="coerce"
                )
                *
                100
            ).round(
                1
            )

            display["Churn Probability"] = (
                display[
                    "Churn Probability"
                ]
                .astype(str)
                +
                "%"
            )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            height=500
        )


    # =========================================================
    # RENDER
    # =========================================================

    def render(
        self,
        df: pd.DataFrame
    ):

        data = self.prepare_data(
            df
        )

        metrics = self.calculate_metrics(
            data
        )

        st.markdown(
            "## 🧩 Customer Segmentation Intelligence"
        )

        st.caption(
            "Interactive analysis of customer groups, "
            "behavioral patterns, predicted revenue, "
            "customer value and churn exposure."
        )

        self._render_kpis(
            metrics
        )

        st.write("")
        st.divider()

        tab1, tab2, tab3, tab4 = (
            st.tabs(
                [
                    "📊 Segment Overview",
                    "🧠 Behavioral Profiles",
                    "💰 Value & Risk",
                    "🎯 Segment Strategy"
                ]
            )
        )

        # =====================================================
        # TAB 1 — OVERVIEW
        # =====================================================

        with tab1:

            distribution = (
                self.segment_distribution(
                    data
                )
            )

            col1, col2 = st.columns(
                2
            )

            with col1:

                bar = px.bar(
                    distribution,

                    x=
                        "Customer Segment",

                    y=
                        "Customers",

                    color=
                        "Customer Segment",

                    text=
                        "Customers",

                    title=
                        "Customers by Segment"
                )

                bar.update_layout(
                    showlegend=False
                )

                st.plotly_chart(
                    bar,
                    use_container_width=True
                )

            with col2:

                donut = px.pie(
                    distribution,

                    names=
                        "Customer Segment",

                    values=
                        "Customers",

                    hole=
                        0.55,

                    title=
                        "Segment Share"
                )

                donut.update_traces(
                    textinfo=
                        "percent+label"
                )

                st.plotly_chart(
                    donut,
                    use_container_width=True
                )

            treemap = self._segment_treemap(
                data
            )

            if treemap is not None:

                st.plotly_chart(
                    treemap,
                    use_container_width=True
                )

            st.dataframe(
                distribution,
                use_container_width=True,
                hide_index=True
            )


        # =====================================================
        # TAB 2 — BEHAVIOR
        # =====================================================

        with tab2:

            st.markdown(
                "### 🧠 Segment Behavioral Profile"
            )

            radar = self._segment_radar(
                data
            )

            if radar is not None:

                st.plotly_chart(
                    radar,
                    use_container_width=True
                )

            profile = self.segment_profile(
                data
            )

            if profile.empty:

                st.info(
                    "Behavioral profile data "
                    "is not available."
                )

            else:

                st.dataframe(
                    profile,
                    use_container_width=True,
                    hide_index=True
                )

                metric_options = [
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

                selected_metric = (
                    st.selectbox(
                        "Select behavior metric",

                        options=
                            metric_options,

                        key=
                            "premium_segment_metric"
                    )
                )

                chart = px.bar(
                    profile,

                    x=
                        "Customer Segment",

                    y=
                        selected_metric,

                    color=
                        selected_metric,

                    text=
                        selected_metric,

                    color_continuous_scale=
                        "Blues",

                    title=
                        (
                            f"Average {selected_metric} "
                            f"by Segment"
                        )
                )

                chart.update_layout(
                    coloraxis_showscale=False
                )

                st.plotly_chart(
                    chart,
                    use_container_width=True
                )

            st.info(
                "Radar values are normalized within the "
                "current customer population. Recency is "
                "reversed so higher values represent more "
                "recent customer engagement."
            )


        # =====================================================
        # TAB 3 — VALUE & RISK
        # =====================================================

        with tab3:

            revenue = self.revenue_contribution(
                data
            )

            if not revenue.empty:

                col1, col2 = st.columns(
                    2
                )

                with col1:

                    chart = px.bar(
                        revenue,

                        x=
                            "Customer Segment",

                        y=
                            "Total Predicted Revenue",

                        color=
                            "Total Predicted Revenue",

                        text=
                            "Total Predicted Revenue",

                        color_continuous_scale=
                            "Viridis",

                        title=
                            "Predicted Revenue by Segment"
                    )

                    chart.update_layout(
                        coloraxis_showscale=False
                    )

                    st.plotly_chart(
                        chart,
                        use_container_width=True
                    )

                with col2:

                    pie = px.pie(
                        revenue,

                        names=
                            "Customer Segment",

                        values=
                            "Total Predicted Revenue",

                        hole=
                            0.55,

                        title=
                            "Revenue Contribution Share"
                    )

                    st.plotly_chart(
                        pie,
                        use_container_width=True
                    )

            st.markdown(
                "### ⚠️ Churn Exposure"
            )

            churn_data = self.churn_exposure(
                data
            )

            if churn_data.empty:

                st.info(
                    "Churn Risk data "
                    "is not available."
                )

            else:

                churn_chart = px.bar(
                    churn_data,

                    x=
                        "Customer Segment",

                    y=
                        "Customers",

                    color=
                        "Churn Risk",

                    barmode=
                        "stack",

                    color_discrete_map={
                        "High":
                            "#ef4444",

                        "Medium":
                            "#f59e0b",

                        "Low":
                            "#10b981"
                    },

                    title=
                        "Churn Risk Composition by Segment"
                )

                st.plotly_chart(
                    churn_chart,
                    use_container_width=True
                )

            st.markdown(
                "### 💎 CLV Composition"
            )

            clv_data = self.clv_composition(
                data
            )

            if clv_data.empty:

                st.info(
                    "CLV Value Band data "
                    "is not available."
                )

            else:

                clv_chart = px.bar(
                    clv_data,

                    x=
                        "Customer Segment",

                    y=
                        "Customers",

                    color=
                        "CLV Value Band",

                    barmode=
                        "stack",

                    color_discrete_map={
                        "High":
                            "#7c3aed",

                        "Medium":
                            "#2563eb",

                        "Low":
                            "#94a3b8"
                    },

                    title=
                        "Customer Value Composition by Segment"
                )

                st.plotly_chart(
                    clv_chart,
                    use_container_width=True
                )

            performance = (
                self.segment_business_performance(
                    data
                )
            )

            st.markdown(
                "### 📋 Segment Business Performance"
            )

            st.dataframe(
                performance,
                use_container_width=True,
                hide_index=True
            )


        # =====================================================
        # TAB 4 — STRATEGY
        # =====================================================

        with tab4:

            strategies = (
                self.segment_actions(
                    data
                )
            )

            st.markdown(
                "### 🎯 Recommended Segment Strategies"
            )

            if strategies.empty:

                st.info(
                    "Segment strategy data "
                    "is unavailable."
                )

            else:

                strategy_types = (
                    strategies[
                        "Strategy"
                    ]
                    .value_counts()
                    .rename_axis(
                        "Strategy"
                    )
                    .reset_index(
                        name="Segments"
                    )
                )

                strategy_chart = px.pie(
                    strategy_types,

                    names=
                        "Strategy",

                    values=
                        "Segments",

                    hole=
                        0.55,

                    title=
                        "Segment Strategy Mix"
                )

                st.plotly_chart(
                    strategy_chart,
                    use_container_width=True
                )

                st.dataframe(
                    strategies,
                    use_container_width=True,
                    hide_index=True
                )

            st.warning(
                "Segment strategies are rule-based "
                "business suggestions derived from "
                "segment churn and value profiles. "
                "They are not ML predictions."
            )

            st.divider()

            st.markdown(
                "### 👥 Segment Customer Explorer"
            )

            segment_options = sorted(
                data[
                    "Customer Segment"
                ]
                .dropna()
                .unique()
                .tolist()
            )

            selected_segment = (
                st.selectbox(
                    "Select Customer Segment",

                    options=
                        segment_options,

                    key=
                        "premium_segment_explorer"
                )
            )

            segment_table = (
                self.segment_customers(
                    data,

                    segment=
                        selected_segment,

                    top_n=
                        100
                )
            )

            self._styled_customer_table(
                segment_table
            )