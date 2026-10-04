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


class CLVIntelligence:

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
            "Predicted 90-Day Revenue",
            "Churn Probability",
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

        if "CLV Value Band" in data.columns:

            data["CLV Value Band"] = (
                data["CLV Value Band"]
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

        # -----------------------------------------------------
        # REVENUE
        # -----------------------------------------------------

        if "Predicted 90-Day Revenue" in data.columns:

            revenue = (
                data[
                    "Predicted 90-Day Revenue"
                ]
                .fillna(0)
                .clip(lower=0)
            )

        else:

            revenue = pd.Series(
                np.zeros(
                    len(data)
                ),
                index=data.index
            )

        total_revenue = float(
            revenue.sum()
        )

        average_revenue = float(
            revenue.mean()
        )

        median_revenue = float(
            revenue.median()
        )

        max_revenue = float(
            revenue.max()
        )

        # -----------------------------------------------------
        # VALUE BANDS
        # -----------------------------------------------------

        if "CLV Value Band" in data.columns:

            value_band = (
                data[
                    "CLV Value Band"
                ]
                .astype(str)
                .str.lower()
            )

            high_value = int(
                (
                    value_band
                    ==
                    "high"
                )
                .sum()
            )

            medium_value = int(
                (
                    value_band
                    ==
                    "medium"
                )
                .sum()
            )

            low_value = int(
                (
                    value_band
                    ==
                    "low"
                )
                .sum()
            )

        else:

            high_value = 0
            medium_value = 0
            low_value = 0

        if total_customers > 0:

            high_value_percentage = (
                high_value
                /
                total_customers
                *
                100
            )

        else:

            high_value_percentage = 0.0

        # -----------------------------------------------------
        # HIGH VALUE REVENUE
        # -----------------------------------------------------

        if "CLV Value Band" in data.columns:

            high_value_mask = (
                data[
                    "CLV Value Band"
                ]
                .astype(str)
                .str.lower()
                .eq("high")
            )

            high_value_revenue = float(
                revenue[
                    high_value_mask
                ]
                .sum()
            )

        else:

            high_value_revenue = 0.0

        # -----------------------------------------------------
        # HIGH VALUE AT RISK
        # -----------------------------------------------------

        if (
            "CLV Value Band" in data.columns
            and
            "Churn Risk" in data.columns
        ):

            high_value_at_risk = int(
                (
                    data[
                        "CLV Value Band"
                    ]
                    .astype(str)
                    .str.lower()
                    .eq("high")
                    &
                    data[
                        "Churn Risk"
                    ]
                    .astype(str)
                    .str.lower()
                    .eq("high")
                )
                .sum()
            )

        else:

            high_value_at_risk = 0

        return {

            "total_customers":
                total_customers,

            "total_predicted_revenue":
                total_revenue,

            "average_predicted_revenue":
                average_revenue,

            "median_predicted_revenue":
                median_revenue,

            "maximum_predicted_revenue":
                max_revenue,

            "high_value_customers":
                high_value,

            "medium_value_customers":
                medium_value,

            "low_value_customers":
                low_value,

            "high_value_percentage":
                high_value_percentage,

            "high_value_revenue":
                high_value_revenue,

            "high_value_at_risk":
                high_value_at_risk
        }


    # =========================================================
    # VALUE BAND DISTRIBUTION
    # =========================================================

    def value_band_distribution(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = self.prepare_data(
            df
        )

        if "CLV Value Band" not in data.columns:

            return pd.DataFrame()

        order = [
            "High",
            "Medium",
            "Low",
            "Unknown"
        ]

        result = (
            data[
                "CLV Value Band"
            ]
            .value_counts()
            .reindex(
                order,
                fill_value=0
            )
            .rename_axis(
                "CLV Value Band"
            )
            .reset_index(
                name="Customers"
            )
        )

        total = max(
            int(
                result[
                    "Customers"
                ]
                .sum()
            ),
            1
        )

        result[
            "Customer Percentage"
        ] = (
            result[
                "Customers"
            ]
            /
            total
            *
            100
        ).round(
            2
        )

        return result


    # =========================================================
    # REVENUE BY VALUE BAND
    # =========================================================

    def revenue_by_value_band(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = self.prepare_data(
            df
        )

        required = [
            "CLV Value Band",
            "Predicted 90-Day Revenue"
        ]

        if any(
            column not in data.columns
            for column in required
        ):

            return pd.DataFrame()

        result = (
            data
            .groupby(
                "CLV Value Band"
            )
            .agg(
                **{
                    "Customers":
                        (
                            CUSTOMER_ID,
                            "nunique"
                        ),

                    "Average Predicted Revenue":
                        (
                            "Predicted 90-Day Revenue",
                            "mean"
                        ),

                    "Total Predicted Revenue":
                        (
                            "Predicted 90-Day Revenue",
                            "sum"
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
            "Average Predicted Revenue",
            "Total Predicted Revenue",
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

                ascending=False
            )
            .reset_index(
                drop=True
            )
        )


    # =========================================================
    # CLV BY SEGMENT
    # =========================================================

    def clv_by_segment(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = self.prepare_data(
            df
        )

        required = [
            "Customer Segment",
            "Predicted 90-Day Revenue"
        ]

        if any(
            column not in data.columns
            for column in required
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

                    "Average Predicted Revenue":
                        (
                            "Predicted 90-Day Revenue",
                            "mean"
                        ),

                    "Median Predicted Revenue":
                        (
                            "Predicted 90-Day Revenue",
                            "median"
                        ),

                    "Total Predicted Revenue":
                        (
                            "Predicted 90-Day Revenue",
                            "sum"
                        )
                }
            )
            .reset_index()
        )

        if "CLV Value Band" in data.columns:

            high_value = (
                data[
                    data[
                        "CLV Value Band"
                    ]
                    .astype(str)
                    .str.lower()
                    .eq("high")
                ]
                .groupby(
                    "Customer Segment"
                )[CUSTOMER_ID]
                .nunique()
                .rename(
                    "High Value Customers"
                )
            )

            result = result.merge(
                high_value,
                on=
                    "Customer Segment",
                how=
                    "left"
            )

            result[
                "High Value Customers"
            ] = (
                result[
                    "High Value Customers"
                ]
                .fillna(0)
                .astype(int)
            )

            result[
                "High Value %"
            ] = (
                result[
                    "High Value Customers"
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

                ascending=False
            )
            .reset_index(
                drop=True
            )
        )


    # =========================================================
    # HIGH VALUE AT RISK
    # =========================================================

    def high_value_at_risk(
        self,
        df: pd.DataFrame,
        top_n: int = 50
    ) -> pd.DataFrame:

        data = self.prepare_data(
            df
        )

        required = [
            "CLV Value Band",
            "Churn Risk"
        ]

        if any(
            column not in data.columns
            for column in required
        ):

            return pd.DataFrame()

        mask = (
            data[
                "CLV Value Band"
            ]
            .astype(str)
            .str.lower()
            .eq("high")
            &
            data[
                "Churn Risk"
            ]
            .astype(str)
            .str.lower()
            .eq("high")
        )

        result = (
            data[
                mask
            ]
            .copy()
        )

        sort_columns = []
        ascending = []

        if "Predicted 90-Day Revenue" in result.columns:

            sort_columns.append(
                "Predicted 90-Day Revenue"
            )

            ascending.append(
                False
            )

        if "Churn Probability" in result.columns:

            sort_columns.append(
                "Churn Probability"
            )

            ascending.append(
                False
            )

        if sort_columns:

            result = (
                result
                .sort_values(
                    by=
                        sort_columns,

                    ascending=
                        ascending
                )
            )

        desired_columns = [
            CUSTOMER_ID,
            "Customer Segment",
            "Predicted 90-Day Revenue",
            "CLV Value Band",
            "Churn Probability",
            "Churn Risk",
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
    # CUSTOMER VALUE RANKING
    # =========================================================

    def customer_value_ranking(
        self,
        df: pd.DataFrame,
        top_n: int = 100
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
            .sort_values(
                by=
                    "Predicted 90-Day Revenue",

                ascending=False
            )
            .head(
                top_n
            )
            .copy()
        )

        result[
            "Customer Value Rank"
        ] = np.arange(
            1,
            len(result) + 1
        )

        desired_columns = [
            "Customer Value Rank",
            CUSTOMER_ID,
            "Customer Segment",
            "Predicted 90-Day Revenue",
            "CLV Value Band",
            "Churn Probability",
            "Churn Risk",
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
    # REVENUE CONCENTRATION
    # =========================================================

    def revenue_concentration(
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

        ranking = (
            data[
                [
                    CUSTOMER_ID,
                    "Predicted 90-Day Revenue"
                ]
            ]
            .dropna()
            .sort_values(
                by=
                    "Predicted 90-Day Revenue",

                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        if ranking.empty:

            return pd.DataFrame()

        ranking[
            "Predicted 90-Day Revenue"
        ] = (
            ranking[
                "Predicted 90-Day Revenue"
            ]
            .clip(
                lower=0
            )
        )

        total_revenue = float(
            ranking[
                "Predicted 90-Day Revenue"
            ]
            .sum()
        )

        customer_count = len(
            ranking
        )

        ranking[
            "Customer Percentile"
        ] = (
            (
                ranking.index
                +
                1
            )
            /
            customer_count
            *
            100
        )

        if total_revenue > 0:

            ranking[
                "Cumulative Revenue %"
            ] = (
                ranking[
                    "Predicted 90-Day Revenue"
                ]
                .cumsum()
                /
                total_revenue
                *
                100
            )

        else:

            ranking[
                "Cumulative Revenue %"
            ] = 0.0

        return ranking


    # =========================================================
    # TOP CUSTOMER SHARE
    # =========================================================

    def top_customer_revenue_share(
        self,
        df: pd.DataFrame,
        top_percentage: float = 20
    ) -> float:

        concentration = (
            self.revenue_concentration(
                df
            )
        )

        if concentration.empty:

            return 0.0

        cutoff = max(
            int(
                np.ceil(
                    len(concentration)
                    *
                    top_percentage
                    /
                    100
                )
            ),
            1
        )

        top_revenue = float(
            concentration[
                "Predicted 90-Day Revenue"
            ]
            .head(
                cutoff
            )
            .sum()
        )

        total_revenue = float(
            concentration[
                "Predicted 90-Day Revenue"
            ]
            .sum()
        )

        if total_revenue <= 0:

            return 0.0

        return float(
            top_revenue
            /
            total_revenue
            *
            100
        )


    # =========================================================
    # KPI CARDS
    # =========================================================

    def _render_kpis(
        self,
        metrics: dict,
        top_20_share: float
    ):

        row1 = st.columns(
            4
        )

        with row1[0]:

            render_color_card(
                title=
                    "Predicted 90-Day Revenue",

                value=
                    (
                        f"{metrics['total_predicted_revenue']:,.0f}"
                    ),

                icon=
                    "💰",

                card_class=
                    "card-green"
            )

        with row1[1]:

            render_color_card(
                title=
                    "Average Customer Value",

                value=
                    (
                        f"{metrics['average_predicted_revenue']:,.0f}"
                    ),

                icon=
                    "📈",

                card_class=
                    "card-blue"
            )

        with row1[2]:

            render_color_card(
                title=
                    "High Value Customers",

                value=
                    (
                        f"{metrics['high_value_customers']:,}"
                    ),

                icon=
                    "💎",

                card_class=
                    "card-purple"
            )

        with row1[3]:

            render_color_card(
                title=
                    "High Value at Risk",

                value=
                    (
                        f"{metrics['high_value_at_risk']:,}"
                    ),

                icon=
                    "🔥",

                card_class=
                    "card-red"
            )

        st.write("")

        row2 = st.columns(
            4
        )

        with row2[0]:

            render_color_card(
                title=
                    "Median Customer Value",

                value=
                    (
                        f"{metrics['median_predicted_revenue']:,.0f}"
                    ),

                icon=
                    "📊",

                card_class=
                    "card-blue"
            )

        with row2[1]:

            render_color_card(
                title=
                    "Maximum Customer Value",

                value=
                    (
                        f"{metrics['maximum_predicted_revenue']:,.0f}"
                    ),

                icon=
                    "🏆",

                card_class=
                    "card-green"
            )

        with row2[2]:

            render_color_card(
                title=
                    "High Value Customer %",

                value=
                    (
                        f"{metrics['high_value_percentage']:.1f}%"
                    ),

                icon=
                    "⭐",

                card_class=
                    "card-purple"
            )

        with row2[3]:

            render_color_card(
                title=
                    "Top 20% Revenue Share",

                value=
                    (
                        f"{top_20_share:.1f}%"
                    ),

                icon=
                    "🎯",

                card_class=
                    "card-red"
            )


    # =========================================================
    # VALUE BAND DONUT
    # =========================================================

    def _value_band_donut(
        self,
        df: pd.DataFrame
    ):

        distribution = (
            self.value_band_distribution(
                df
            )
        )

        if distribution.empty:

            return None

        distribution = (
            distribution[
                distribution[
                    "Customers"
                ]
                >
                0
            ]
        )

        figure = px.pie(
            distribution,

            names=
                "CLV Value Band",

            values=
                "Customers",

            hole=
                0.58,

            color=
                "CLV Value Band",

            color_discrete_map={
                "High":
                    "#7c3aed",

                "Medium":
                    "#2563eb",

                "Low":
                    "#94a3b8",

                "Unknown":
                    "#cbd5e1"
            },

            title=
                "Customer Value Mix"
        )

        figure.update_traces(
            textposition=
                "inside",

            textinfo=
                "percent+label"
        )

        figure.update_layout(
            height=390,
            legend_title_text=""
        )

        return figure


    # =========================================================
    # REVENUE GAUGE
    # =========================================================

    def _high_value_share_gauge(
        self,
        metrics: dict
    ):

        value = float(
            metrics[
                "high_value_percentage"
            ]
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
                        "High Value Customer Share"
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
                                20
                            ],
                            "color":
                                "#e2e8f0"
                        },

                        {
                            "range": [
                                20,
                                50
                            ],
                            "color":
                                "#dbeafe"
                        },

                        {
                            "range": [
                                50,
                                100
                            ],
                            "color":
                                "#ede9fe"
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
    # VALUE × CHURN SCATTER
    # =========================================================

    def _value_churn_scatter(
        self,
        df: pd.DataFrame
    ):

        data = self.prepare_data(
            df
        )

        required = [
            "Predicted 90-Day Revenue",
            "Churn Probability"
        ]

        if any(
            column not in data.columns
            for column in required
        ):

            return None

        plot_data = (
            data
            .dropna(
                subset=[
                    "Predicted 90-Day Revenue",
                    "Churn Probability"
                ]
            )
            .copy()
        )

        if plot_data.empty:

            return None

        revenue_boundary = float(
            plot_data[
                "Predicted 90-Day Revenue"
            ]
            .median()
        )

        churn_boundary = 0.70

        hover_data = [
            CUSTOMER_ID
        ]

        for column in [
            "Customer Segment",
            "CLV Value Band",
            "Churn Risk",
            "Top Recommended Product"
        ]:

            if column in plot_data.columns:

                hover_data.append(
                    column
                )

        figure = px.scatter(
            plot_data,

            x=
                "Predicted 90-Day Revenue",

            y=
                "Churn Probability",

            color=(
                "CLV Value Band"
                if
                "CLV Value Band"
                in plot_data.columns
                else
                None
            ),

            color_discrete_map={
                "High":
                    "#7c3aed",

                "Medium":
                    "#2563eb",

                "Low":
                    "#94a3b8"
            },

            hover_data=
                hover_data,

            title=
                "Customer Value × Churn Probability"
        )

        figure.add_vline(
            x=
                revenue_boundary,

            line_dash=
                "dash",

            line_color=
                "#64748b"
        )

        figure.add_hline(
            y=
                churn_boundary,

            line_dash=
                "dash",

            line_color=
                "#ef4444"
        )

        figure.add_annotation(
            x=
                revenue_boundary,

            y=
                0.95,

            text=
                "Higher Customer Value →",

            showarrow=
                False,

            xshift=
                75
        )

        figure.add_annotation(
            x=
                revenue_boundary,

            y=
                churn_boundary,

            text=
                "High Value + High Risk",

            showarrow=
                False,

            xshift=
                100,

            yshift=
                35
        )

        figure.update_layout(
            height=520
        )

        return figure


    # =========================================================
    # STYLED TABLE
    # =========================================================

    def _style_customer_table(
        self,
        table: pd.DataFrame
    ):

        if table.empty:

            st.info(
                "No customer records available."
            )

            return

        display = table.copy()

        if "CLV Value Band" in display.columns:

            display[
                "CLV Value Band"
            ] = display[
                "CLV Value Band"
            ].apply(
                format_clv_badge
            )

        if "Churn Risk" in display.columns:

            display[
                "Churn Risk"
            ] = display[
                "Churn Risk"
            ].apply(
                format_risk_badge
            )

        if "Churn Probability" in display.columns:

            display[
                "Churn Probability"
            ] = (
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
            "Predicted 90-Day Revenue"
            not in data.columns
        ):

            st.warning(
                "Predicted 90-Day Revenue "
                "data is not available."
            )

            return

        metrics = self.calculate_metrics(
            data
        )

        top_20_share = (
            self.top_customer_revenue_share(
                data,
                top_percentage=20
            )
        )

        st.markdown(
            "## 💰 Customer Value Intelligence"
        )

        st.caption(
            "Interactive analysis of predicted 90-day "
            "customer revenue, value concentration, "
            "customer segments and churn exposure."
        )

        # =====================================================
        # KPI CARDS
        # =====================================================

        self._render_kpis(
            metrics,
            top_20_share
        )

        st.write("")
        st.divider()

        # =====================================================
        # TABS
        # =====================================================

        tab1, tab2, tab3, tab4 = (
            st.tabs(
                [
                    "💎 Value Overview",
                    "🧩 Segment Value",
                    "⚠️ Value × Churn",
                    "🏆 Customer Ranking"
                ]
            )
        )

        # =====================================================
        # TAB 1 — VALUE OVERVIEW
        # =====================================================

        with tab1:

            col1, col2 = st.columns(
                2
            )

            with col1:

                donut = (
                    self._value_band_donut(
                        data
                    )
                )

                if donut is not None:

                    st.plotly_chart(
                        donut,
                        use_container_width=True
                    )

            with col2:

                gauge = (
                    self._high_value_share_gauge(
                        metrics
                    )
                )

                st.plotly_chart(
                    gauge,
                    use_container_width=True
                )

            st.markdown(
                "### 📈 Predicted Revenue Distribution"
            )

            histogram_data = (
                data[
                    data[
                        "Predicted 90-Day Revenue"
                    ]
                    .notna()
                ]
            )

            histogram = px.histogram(
                histogram_data,

                x=
                    "Predicted 90-Day Revenue",

                nbins=
                    40,

                color=(
                    "CLV Value Band"
                    if
                    "CLV Value Band"
                    in histogram_data.columns
                    else
                    None
                ),

                color_discrete_map={
                    "High":
                        "#7c3aed",

                    "Medium":
                        "#2563eb",

                    "Low":
                        "#94a3b8"
                },

                title=
                    "Predicted 90-Day Revenue Distribution"
            )

            histogram.update_layout(
                xaxis_title=
                    "Predicted 90-Day Revenue",

                yaxis_title=
                    "Customers",

                height=
                    450
            )

            st.plotly_chart(
                histogram,
                use_container_width=True
            )

            band_revenue = (
                self.revenue_by_value_band(
                    data
                )
            )

            if not band_revenue.empty:

                st.markdown(
                    "### 💵 Revenue Contribution by Value Band"
                )

                col3, col4 = st.columns(
                    2
                )

                with col3:

                    chart = px.bar(
                        band_revenue,

                        x=
                            "CLV Value Band",

                        y=
                            "Total Predicted Revenue",

                        color=
                            "CLV Value Band",

                        text=
                            "Total Predicted Revenue",

                        color_discrete_map={
                            "High":
                                "#7c3aed",

                            "Medium":
                                "#2563eb",

                            "Low":
                                "#94a3b8"
                        },

                        title=
                            "Revenue by CLV Band"
                    )

                    st.plotly_chart(
                        chart,
                        use_container_width=True
                    )

                with col4:

                    pie = px.pie(
                        band_revenue,

                        names=
                            "CLV Value Band",

                        values=
                            "Total Predicted Revenue",

                        hole=
                            0.55,

                        color=
                            "CLV Value Band",

                        color_discrete_map={
                            "High":
                                "#7c3aed",

                            "Medium":
                                "#2563eb",

                            "Low":
                                "#94a3b8"
                        },

                        title=
                            "Revenue Contribution Share"
                    )

                    st.plotly_chart(
                        pie,
                        use_container_width=True
                    )

                st.dataframe(
                    band_revenue,
                    use_container_width=True,
                    hide_index=True
                )


        # =====================================================
        # TAB 2 — SEGMENT VALUE
        # =====================================================

        with tab2:

            segment_clv = (
                self.clv_by_segment(
                    data
                )
            )

            if segment_clv.empty:

                st.info(
                    "Customer Segment data "
                    "is not available."
                )

            else:

                col1, col2 = st.columns(
                    2
                )

                with col1:

                    average_chart = px.bar(
                        segment_clv,

                        x=
                            "Customer Segment",

                        y=
                            "Average Predicted Revenue",

                        color=
                            "Average Predicted Revenue",

                        text=
                            "Average Predicted Revenue",

                        color_continuous_scale=
                            "Blues",

                        title=
                            "Average Customer Value by Segment"
                    )

                    average_chart.update_layout(
                        coloraxis_showscale=False
                    )

                    st.plotly_chart(
                        average_chart,
                        use_container_width=True
                    )

                with col2:

                    total_chart = px.bar(
                        segment_clv,

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
                            "Total Predicted Revenue by Segment"
                    )

                    total_chart.update_layout(
                        coloraxis_showscale=False
                    )

                    st.plotly_chart(
                        total_chart,
                        use_container_width=True
                    )

                st.dataframe(
                    segment_clv,
                    use_container_width=True,
                    hide_index=True
                )


        # =====================================================
        # TAB 3 — VALUE × CHURN
        # =====================================================

        with tab3:

            scatter = (
                self._value_churn_scatter(
                    data
                )
            )

            if scatter is None:

                st.info(
                    "Churn Probability data "
                    "is required for this view."
                )

            else:

                st.plotly_chart(
                    scatter,
                    use_container_width=True
                )

                st.info(
                    "The upper-right area contains customers "
                    "with relatively high predicted value "
                    "and high churn probability."
                )

            st.markdown(
                "### 🔥 High-Value Customers at Risk"
            )

            at_risk = (
                self.high_value_at_risk(
                    data,
                    top_n=50
                )
            )

            if at_risk.empty:

                st.success(
                    "No High CLV + High Churn Risk "
                    "customers were found for the "
                    "current filter selection."
                )

            else:

                self._style_customer_table(
                    at_risk
                )


        # =====================================================
        # TAB 4 — CUSTOMER RANKING
        # =====================================================

        with tab4:

            st.markdown(
                "### 🏆 Highest Predicted-Value Customers"
            )

            ranking = (
                self.customer_value_ranking(
                    data,
                    top_n=100
                )
            )

            self._style_customer_table(
                ranking
            )

            st.divider()

            st.markdown(
                "### 📊 Revenue Concentration"
            )

            concentration = (
                self.revenue_concentration(
                    data
                )
            )

            if concentration.empty:

                st.info(
                    "Revenue concentration data "
                    "is not available."
                )

            else:

                concentration_chart = (
                    go.Figure()
                )

                concentration_chart.add_trace(
                    go.Scatter(
                        x=
                            concentration[
                                "Customer Percentile"
                            ],

                        y=
                            concentration[
                                "Cumulative Revenue %"
                            ],

                        mode=
                            "lines",

                        name=
                            "Cumulative Revenue",

                        line={
                            "width":
                                4,

                            "color":
                                "#7c3aed"
                        },

                        fill=
                            "tozeroy",

                        fillcolor=
                            "rgba(124,58,237,0.10)"
                    )
                )

                # 20% marker
                concentration_chart.add_vline(
                    x=
                        20,

                    line_dash=
                        "dash",

                    line_color=
                        "#ef4444"
                )

                concentration_chart.add_annotation(
                    x=
                        20,

                    y=
                        top_20_share,

                    text=
                        (
                            f"Top 20% → "
                            f"{top_20_share:.1f}% revenue"
                        ),

                    showarrow=
                        True,

                    arrowhead=
                        2
                )

                # Equality/reference line
                concentration_chart.add_trace(
                    go.Scatter(
                        x=[
                            0,
                            100
                        ],

                        y=[
                            0,
                            100
                        ],

                        mode=
                            "lines",

                        name=
                            "Equal Distribution",

                        line={
                            "dash":
                                "dot",

                            "color":
                                "#94a3b8"
                        }
                    )
                )

                concentration_chart.update_layout(
                    title=
                        (
                            "Cumulative Predicted Revenue "
                            "by Customer Percentile"
                        ),

                    xaxis_title=
                        "Top Customers Included (%)",

                    yaxis_title=
                        "Cumulative Predicted Revenue (%)",

                    xaxis_range=[
                        0,
                        100
                    ],

                    yaxis_range=[
                        0,
                        100
                    ],

                    height=
                        500
                )

                st.plotly_chart(
                    concentration_chart,
                    use_container_width=True
                )

                st.success(
                    (
                        f"🎯 The top 20% of customers "
                        f"represent approximately "
                        f"**{top_20_share:.1f}%** of "
                        f"predicted 90-day revenue "
                        f"for the current filtered population."
                    )
                )

            st.warning(
                "Predicted 90-Day Revenue is a fixed-horizon "
                "future revenue prediction. It should not "
                "be interpreted as literal lifetime revenue."
            )