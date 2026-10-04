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

        numeric_columns = [
            "Churn Probability",
            "Predicted 90-Day Revenue",
            "Recency",
            "Frequency",
            "Monetary",
            "TotalItems",
            "AverageOrderValue",
            "Tenure"
        ]

        for column in numeric_columns:

            if column in data.columns:

                data[column] = pd.to_numeric(
                    data[column],
                    errors="coerce"
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
            data[
                CUSTOMER_ID
            ].nunique()
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
                .fillna(
                    "Unknown"
                )
                .astype(str)
                .str.lower()
            )

            high_risk = int(
                (
                    normalized_risk
                    ==
                    "high"
                )
                .sum()
            )

            medium_risk = int(
                (
                    normalized_risk
                    ==
                    "medium"
                )
                .sum()
            )

            low_risk = int(
                (
                    normalized_risk
                    ==
                    "low"
                )
                .sum()
            )

        else:

            high_risk = 0
            medium_risk = 0
            low_risk = 0

        # -----------------------------------------------------
        # High risk percentage
        # -----------------------------------------------------

        if total_customers > 0:

            high_risk_percentage = (
                high_risk
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
            "Churn Risk"
            in data.columns
            and
            "Predicted 90-Day Revenue"
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
        # High-value at risk
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
                high_risk,

            "medium_risk_customers":
                medium_risk,

            "low_risk_customers":
                low_risk,

            "high_risk_percentage":
                high_risk_percentage,

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

        data = self.prepare_data(
            df
        )

        if (
            "Churn Risk"
            not in data.columns
        ):

            return pd.DataFrame()

        order = [
            "High",
            "Medium",
            "Low"
        ]

        return (
            data[
                "Churn Risk"
            ]
            .value_counts()
            .reindex(
                order,
                fill_value=0
            )
            .rename_axis(
                "Churn Risk"
            )
            .reset_index(
                name="Customers"
            )
        )


    # =========================================================
    # CHURN BY SEGMENT
    # =========================================================

    def churn_by_segment(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = self.prepare_data(
            df
        )

        required = [
            "Customer Segment",
            "Churn Probability"
        ]

        if any(
            column not in data.columns
            for column in required
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
            if
            "Churn Risk"
            in data.columns
            else
            0
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

        return (
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


    # =========================================================
    # REVENUE AT RISK BY SEGMENT
    # =========================================================

    def revenue_at_risk_by_segment(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = self.prepare_data(
            df
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

            return pd.DataFrame()

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

        return (
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


    # =========================================================
    # HIGH VALUE AT RISK
    # =========================================================

    def high_value_customers_at_risk(
        self,
        df: pd.DataFrame,
        top_n: int = 50
    ) -> pd.DataFrame:

        data = self.prepare_data(
            df
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
            and
            "Churn Probability"
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
            for column
            in desired_columns
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
    # RETENTION PRIORITY
    # =========================================================

    def retention_priority_customers(
        self,
        df: pd.DataFrame,
        top_n: int = 100
    ) -> pd.DataFrame:

        data = self.prepare_data(
            df
        )

        if (
            "Churn Probability"
            not in data.columns
        ):

            return pd.DataFrame()

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
        ).round(
            4
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
            for column
            in desired_columns
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
    # KPI CARDS
    # =========================================================

    def _render_kpis(
        self,
        metrics: dict
    ):

        row1 = st.columns(
            4
        )

        with row1[0]:

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

        with row1[1]:

            render_color_card(
                title=
                    "Avg Churn Probability",

                value=
                    (
                        f"{metrics['average_churn_probability'] * 100:.1f}%"
                    ),

                icon=
                    "📊",

                card_class=
                    "card-purple"
            )

        with row1[2]:

            render_color_card(
                title=
                    "High Risk Customers",

                value=
                    f"{metrics['high_risk_customers']:,}",

                icon=
                    "🔴",

                card_class=
                    "card-red"
            )

        with row1[3]:

            render_color_card(
                title=
                    "High Risk %",

                value=
                    (
                        f"{metrics['high_risk_percentage']:.1f}%"
                    ),

                icon=
                    "⚠️",

                card_class=
                    "card-red"
            )

        st.write("")

        row2 = st.columns(
            3
        )

        with row2[0]:

            render_color_card(
                title=
                    "Revenue at Risk",

                value=
                    (
                        f"{metrics['revenue_at_risk']:,.0f}"
                    ),

                icon=
                    "📉",

                card_class=
                    "card-red"
            )

        with row2[1]:

            render_color_card(
                title=
                    "High-Value Customers at Risk",

                value=
                    (
                        f"{metrics['high_value_at_risk']:,}"
                    ),

                icon=
                    "💎",

                card_class=
                    "card-purple"
            )

        with row2[2]:

            render_color_card(
                title=
                    "Medium Risk Customers",

                value=
                    (
                        f"{metrics['medium_risk_customers']:,}"
                    ),

                icon=
                    "🟠",

                card_class=
                    "card-blue"
            )


    # =========================================================
    # RISK DONUT
    # =========================================================

    def _risk_donut(
        self,
        df: pd.DataFrame
    ):

        distribution = (
            self.risk_distribution(
                df
            )
        )

        if distribution.empty:

            return None

        figure = px.pie(
            distribution,

            names=
                "Churn Risk",

            values=
                "Customers",

            hole=
                0.58,

            color=
                "Churn Risk",

            color_discrete_map={
                "High":
                    "#ef4444",

                "Medium":
                    "#f59e0b",

                "Low":
                    "#10b981"
            },

            title=
                "Customer Churn Risk"
        )

        figure.update_traces(
            textposition=
                "inside",

            textinfo=
                "percent+label"
        )

        figure.update_layout(
            height=380,
            legend_title_text=""
        )

        return figure


    # =========================================================
    # CHURN GAUGE
    # =========================================================

    def _churn_gauge(
        self,
        metrics: dict
    ):

        value = (
            metrics[
                "average_churn_probability"
            ]
            *
            100
        )

        figure = go.Figure(
            go.Indicator(
                mode=
                    "gauge+number",

                value=
                    value,

                number={
                    "suffix":
                        "%"
                },

                title={
                    "text":
                        "Average Churn Probability"
                },

                gauge={

                    "axis": {
                        "range": [
                            0,
                            100
                        ]
                    },

                    "bar": {
                        "color":
                            "#7c3aed"
                    },

                    "steps": [

                        {
                            "range": [
                                0,
                                30
                            ],

                            "color":
                                "#dcfce7"
                        },

                        {
                            "range": [
                                30,
                                70
                            ],

                            "color":
                                "#fef3c7"
                        },

                        {
                            "range": [
                                70,
                                100
                            ],

                            "color":
                                "#fee2e2"
                        }
                    ]
                }
            )
        )

        figure.update_layout(
            height=330,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            )
        )

        return figure


    # =========================================================
    # BEHAVIOR SCATTER
    # =========================================================

    def _behavior_scatter(
        self,
        data: pd.DataFrame,
        feature: str
    ):

        plot_data = (
            data[
                data[
                    feature
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

        if plot_data.empty:

            return None

        hover_columns = [
            CUSTOMER_ID
        ]

        for column in [
            "Customer Segment",
            "Churn Risk",
            "CLV Value Band",
            "Predicted 90-Day Revenue"
        ]:

            if column in plot_data.columns:

                hover_columns.append(
                    column
                )

        figure = px.scatter(
            plot_data,

            x=
                feature,

            y=
                "Churn Probability",

            color=(
                "Churn Risk"
                if
                "Churn Risk"
                in plot_data.columns
                else
                None
            ),

            hover_data=
                hover_columns,

            color_discrete_map={
                "High":
                    "#ef4444",

                "Medium":
                    "#f59e0b",

                "Low":
                    "#10b981"
            },

            title=
                (
                    f"{feature} vs "
                    f"Churn Probability"
                )
        )

        figure.update_layout(
            height=480
        )

        return figure


    # =========================================================
    # STYLED HIGH-RISK TABLE
    # =========================================================

    def _styled_high_risk_table(
        self,
        df: pd.DataFrame
    ):

        data = self.prepare_data(
            df
        )

        columns = [
            column
            for column in [
                CUSTOMER_ID,
                "Customer Segment",
                "Churn Probability",
                "Churn Risk",
                "Predicted 90-Day Revenue",
                "CLV Value Band",
                "Top Recommended Product"
            ]
            if column in data.columns
        ]

        if (
            "Churn Probability"
            not in data.columns
        ):

            return

        table = (
            data[
                columns
            ]
            .sort_values(
                by=
                    "Churn Probability",

                ascending=
                    False
            )
            .head(
                50
            )
            .copy()
        )

        if (
            "Churn Probability"
            in table.columns
        ):

            table[
                "Churn Probability"
            ] = (
                table[
                    "Churn Probability"
                ]
                *
                100
            ).round(
                1
            )

            table[
                "Churn Probability"
            ] = (
                table[
                    "Churn Probability"
                ]
                .astype(str)
                +
                "%"
            )

        if (
            "Churn Risk"
            in table.columns
        ):

            table[
                "Churn Risk"
            ] = table[
                "Churn Risk"
            ].apply(
                format_risk_badge
            )

        if (
            "CLV Value Band"
            in table.columns
        ):

            table[
                "CLV Value Band"
            ] = table[
                "CLV Value Band"
            ].apply(
                format_clv_badge
            )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            height=480
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

        if (
            "Churn Probability"
            not in data.columns
        ):

            st.warning(
                "Churn Probability data is not available."
            )

            return

        metrics = self.calculate_metrics(
            data
        )

        # =====================================================
        # PAGE TITLE
        # =====================================================

        st.markdown(
            "## ⚠️ Churn Intelligence"
        )

        st.caption(
            "Interactive customer-retention intelligence "
            "combining churn probability, behavioural "
            "patterns, customer value and revenue risk."
        )

        # =====================================================
        # KPI CARDS
        # =====================================================

        self._render_kpis(
            metrics
        )

        st.write("")
        st.divider()

        # =====================================================
        # TABS
        # =====================================================

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📊 Risk Overview",
                "🧠 Behaviour Drivers",
                "💰 Revenue Risk",
                "🎯 Retention Priority"
            ]
        )

        # =====================================================
        # TAB 1
        # =====================================================

        with tab1:

            col1, col2 = st.columns(
                2
            )

            with col1:

                risk_donut = (
                    self._risk_donut(
                        data
                    )
                )

                if risk_donut is not None:

                    st.plotly_chart(
                        risk_donut,
                        use_container_width=True
                    )

            with col2:

                churn_gauge = (
                    self._churn_gauge(
                        metrics
                    )
                )

                st.plotly_chart(
                    churn_gauge,
                    use_container_width=True
                )

            st.markdown(
                "### 🧩 Churn by Customer Segment"
            )

            segment_data = (
                self.churn_by_segment(
                    data
                )
            )

            if segment_data.empty:

                st.info(
                    "Segment-level churn data "
                    "is not available."
                )

            else:

                col3, col4 = st.columns(
                    2
                )

                with col3:

                    figure = px.bar(
                        segment_data,

                        x=
                            "Customer Segment",

                        y=
                            "Average Churn Probability",

                        color=
                            "Average Churn Probability",

                        text=
                            "Average Churn Probability",

                        color_continuous_scale=
                            "Reds",

                        title=
                            "Average Churn Probability by Segment"
                    )

                    figure.update_layout(
                        coloraxis_showscale=False
                    )

                    st.plotly_chart(
                        figure,
                        use_container_width=True
                    )

                with col4:

                    figure = px.bar(
                        segment_data,

                        x=
                            "Customer Segment",

                        y=
                            "High Risk Customers",

                        color=
                            "High Risk Customers",

                        text=
                            "High Risk Customers",

                        color_continuous_scale=
                            "Oranges",

                        title=
                            "High-Risk Customers by Segment"
                    )

                    figure.update_layout(
                        coloraxis_showscale=False
                    )

                    st.plotly_chart(
                        figure,
                        use_container_width=True
                    )

                st.dataframe(
                    segment_data,
                    use_container_width=True,
                    hide_index=True
                )

        # =====================================================
        # TAB 2 — BEHAVIOUR
        # =====================================================

        with tab2:

            st.markdown(
                "### 🧠 Customer Behaviour vs Churn"
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

            if not behavior_options:

                st.info(
                    "Customer behavioral features "
                    "are not available."
                )

            else:

                selected_feature = (
                    st.selectbox(
                        "Select customer behaviour feature",

                        options=
                            behavior_options,

                        key=
                            "premium_churn_behavior"
                    )
                )

                scatter = (
                    self._behavior_scatter(
                        data,
                        selected_feature
                    )
                )

                if scatter is not None:

                    st.plotly_chart(
                        scatter,
                        use_container_width=True
                    )

                st.info(
                    "Use hover to inspect individual customers. "
                    "The chart shows association with churn "
                    "probability, not causal impact."
                )

                # ---------------------------------------------
                # Distribution by churn risk
                # ---------------------------------------------

                if (
                    "Churn Risk"
                    in data.columns
                ):

                    box_data = (
                        data[
                            data[
                                selected_feature
                            ]
                            .notna()
                        ]
                    )

                    box_chart = px.box(
                        box_data,

                        x=
                            "Churn Risk",

                        y=
                            selected_feature,

                        color=
                            "Churn Risk",

                        points=
                            "outliers",

                        color_discrete_map={
                            "High":
                                "#ef4444",

                            "Medium":
                                "#f59e0b",

                            "Low":
                                "#10b981"
                        },

                        title=
                            (
                                f"{selected_feature} Distribution "
                                f"by Churn Risk"
                            )
                    )

                    st.plotly_chart(
                        box_chart,
                        use_container_width=True
                    )

        # =====================================================
        # TAB 3 — REVENUE RISK
        # =====================================================

        with tab3:

            st.markdown(
                "### 💰 Revenue at Risk"
            )

            risk_revenue = (
                self.revenue_at_risk_by_segment(
                    data
                )
            )

            if risk_revenue.empty:

                st.info(
                    "Revenue-at-risk analysis "
                    "is not available."
                )

            else:

                figure = px.bar(
                    risk_revenue,

                    x=
                        "Customer Segment",

                    y=
                        "Revenue at Risk",

                    color=
                        "Revenue at Risk",

                    text=
                        "Revenue at Risk",

                    hover_data=[
                        "High Risk Customers"
                    ],

                    color_continuous_scale=
                        "Reds",

                    title=
                        "Predicted Revenue at Risk by Segment"
                )

                figure.update_layout(
                    coloraxis_showscale=False
                )

                st.plotly_chart(
                    figure,
                    use_container_width=True
                )

                st.dataframe(
                    risk_revenue,
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown(
                "### 💎 High-Value Customers at Risk"
            )

            high_value_risk = (
                self.high_value_customers_at_risk(
                    data,
                    top_n=50
                )
            )

            if high_value_risk.empty:

                st.success(
                    "No High CLV + High Churn Risk customers "
                    "were found for the current filters."
                )

            else:

                display = (
                    high_value_risk.copy()
                )

                if (
                    "Churn Risk"
                    in display.columns
                ):

                    display[
                        "Churn Risk"
                    ] = display[
                        "Churn Risk"
                    ].apply(
                        format_risk_badge
                    )

                if (
                    "CLV Value Band"
                    in display.columns
                ):

                    display[
                        "CLV Value Band"
                    ] = display[
                        "CLV Value Band"
                    ].apply(
                        format_clv_badge
                    )

                st.dataframe(
                    display,
                    use_container_width=True,
                    hide_index=True
                )

        # =====================================================
        # TAB 4 — RETENTION
        # =====================================================

        with tab4:

            st.markdown(
                "### 🎯 Retention Priority Customers"
            )

            st.caption(
                "Retention Priority Score = "
                "70% churn probability + "
                "30% normalized predicted 90-day revenue."
            )

            retention = (
                self.retention_priority_customers(
                    data,
                    top_n=100
                )
            )

            if retention.empty:

                st.info(
                    "Retention priority data "
                    "is not available."
                )

            else:

                display = (
                    retention.copy()
                )

                if (
                    "Churn Risk"
                    in display.columns
                ):

                    display[
                        "Churn Risk"
                    ] = display[
                        "Churn Risk"
                    ].apply(
                        format_risk_badge
                    )

                if (
                    "CLV Value Band"
                    in display.columns
                ):

                    display[
                        "CLV Value Band"
                    ] = display[
                        "CLV Value Band"
                    ].apply(
                        format_clv_badge
                    )

                if (
                    "Churn Probability"
                    in display.columns
                ):

                    display[
                        "Churn Probability"
                    ] = (
                        display[
                            "Churn Probability"
                        ]
                        *
                        100
                    ).round(
                        1
                    )

                    display[
                        "Churn Probability"
                    ] = (
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

            st.warning(
                "Retention Priority Score is a "
                "business ranking rule. It is not "
                "another trained machine-learning model."
            )

            st.markdown(
                "### 🔥 Highest Churn Risk Customers"
            )

            self._styled_high_risk_table(
                data
            )