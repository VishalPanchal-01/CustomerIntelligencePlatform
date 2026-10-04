import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.ui import (
    render_color_card,
    format_priority_badge,
    format_risk_badge,
    format_clv_badge
)


CUSTOMER_ID = "Customer ID"


class ExecutiveOverview:

    # =========================================================
    # PRIORITY CLASSIFICATION
    # =========================================================

    def add_customer_priority(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = df.copy()

        if (
            "Churn Risk"
            not in data.columns
        ):

            data[
                "Churn Risk"
            ] = "Unknown"

        if (
            "CLV Value Band"
            not in data.columns
        ):

            data[
                "CLV Value Band"
            ] = "Unknown"

        data[
            "Churn Risk"
        ] = (
            data[
                "Churn Risk"
            ]
            .fillna(
                "Unknown"
            )
            .astype(str)
        )

        data[
            "CLV Value Band"
        ] = (
            data[
                "CLV Value Band"
            ]
            .fillna(
                "Unknown"
            )
            .astype(str)
        )

        data[
            "Customer Priority"
        ] = data.apply(
            self._classify_priority,
            axis=1
        )

        data[
            "Recommended Business Action"
        ] = data.apply(
            self._business_action,
            axis=1
        )

        return data

    # =========================================================
    # PRIORITY RULE
    # =========================================================

    def _classify_priority(
        self,
        row: pd.Series
    ) -> str:

        churn = str(
            row.get(
                "Churn Risk",
                "Unknown"
            )
        ).strip().lower()

        value = str(
            row.get(
                "CLV Value Band",
                "Unknown"
            )
        ).strip().lower()

        if (
            churn == "high"
            and
            value == "high"
        ):

            return "Critical"

        if (
            churn == "medium"
            and
            value == "high"
        ):

            return "High"

        if (
            churn == "high"
            and
            value == "medium"
        ):

            return "High"

        if (
            churn == "low"
            and
            value == "high"
        ):

            return "Medium"

        if (
            churn == "medium"
            and
            value == "medium"
        ):

            return "Medium"

        if (
            churn == "high"
            and
            value == "low"
        ):

            return "Medium"

        if (
            churn == "unknown"
            or
            value == "unknown"
        ):

            return "Unknown"

        return "Low"

    # =========================================================
    # BUSINESS ACTION
    # =========================================================

    def _business_action(
        self,
        row: pd.Series
    ) -> str:

        priority = (
            row.get(
                "Customer Priority",
                "Unknown"
            )
        )

        if priority == "Critical":

            return (
                "Immediate retention action with "
                "personalized engagement."
            )

        if priority == "High":

            return (
                "Prioritize retention campaign and "
                "monitor customer activity."
            )

        if priority == "Medium":

            return (
                "Maintain targeted engagement and "
                "personalized recommendations."
            )

        if priority == "Low":

            return (
                "Continue regular engagement and "
                "monitor future behaviour."
            )

        return (
            "Insufficient intelligence for "
            "priority assignment."
        )

    # =========================================================
    # METRICS
    # =========================================================

    def calculate_metrics(
        self,
        df: pd.DataFrame
    ) -> dict:

        data = (
            self.add_customer_priority(
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
        # REVENUE
        # -----------------------------------------------------

        if (
            "Predicted 90-Day Revenue"
            in data.columns
        ):

            revenue = (
                pd.to_numeric(
                    data[
                        "Predicted 90-Day Revenue"
                    ],
                    errors="coerce"
                )
                .fillna(
                    0
                )
                .clip(
                    lower=0
                )
            )

        else:

            revenue = pd.Series(
                np.zeros(
                    len(data)
                ),
                index=data.index
            )

        total_predicted_revenue = float(
            revenue.sum()
        )

        average_predicted_revenue = float(
            revenue.mean()
        )

        # -----------------------------------------------------
        # HIGH RISK
        # -----------------------------------------------------

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

        high_risk_customers = int(
            high_risk_mask.sum()
        )

        revenue_at_risk = float(
            revenue[
                high_risk_mask
            ]
            .sum()
        )

        if total_predicted_revenue > 0:

            revenue_at_risk_percentage = (
                revenue_at_risk
                /
                total_predicted_revenue
                *
                100
            )

        else:

            revenue_at_risk_percentage = 0.0

        # -----------------------------------------------------
        # PRIORITY
        # -----------------------------------------------------

        critical_mask = (
            data[
                "Customer Priority"
            ]
            ==
            "Critical"
        )

        critical_customers = int(
            critical_mask.sum()
        )

        high_priority_customers = int(
            (
                data[
                    "Customer Priority"
                ]
                ==
                "High"
            )
            .sum()
        )

        critical_revenue = float(
            revenue[
                critical_mask
            ]
            .sum()
        )

        # -----------------------------------------------------
        # HIGH VALUE
        # -----------------------------------------------------

        high_value_customers = int(
            (
                data[
                    "CLV Value Band"
                ]
                .astype(str)
                .str.lower()
                .eq(
                    "high"
                )
            )
            .sum()
        )

        # -----------------------------------------------------
        # CHURN PROBABILITY
        # -----------------------------------------------------

        if (
            "Churn Probability"
            in data.columns
        ):

            average_churn_probability = float(
                pd.to_numeric(
                    data[
                        "Churn Probability"
                    ],
                    errors="coerce"
                )
                .mean()
            )

        else:

            average_churn_probability = 0.0

        return {

            "total_customers":
                total_customers,

            "high_risk_customers":
                high_risk_customers,

            "high_value_customers":
                high_value_customers,

            "critical_customers":
                critical_customers,

            "high_priority_customers":
                high_priority_customers,

            "total_predicted_revenue":
                total_predicted_revenue,

            "average_predicted_revenue":
                average_predicted_revenue,

            "revenue_at_risk":
                revenue_at_risk,

            "revenue_at_risk_percentage":
                revenue_at_risk_percentage,

            "critical_customer_revenue":
                critical_revenue,

            "average_churn_probability":
                average_churn_probability
        }

    # =========================================================
    # PRIORITY DISTRIBUTION
    # =========================================================

    def priority_distribution(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = (
            self.add_customer_priority(
                df
            )
        )

        order = [
            "Critical",
            "High",
            "Medium",
            "Low",
            "Unknown"
        ]

        result = (
            data[
                "Customer Priority"
            ]
            .value_counts()
            .reindex(
                order,
                fill_value=0
            )
            .rename_axis(
                "Customer Priority"
            )
            .reset_index(
                name="Customers"
            )
        )

        return result

    # =========================================================
    # SEGMENT PERFORMANCE
    # =========================================================

    def segment_performance(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = (
            self.add_customer_priority(
                df
            )
        )

        if (
            "Customer Segment"
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
        )

        data[
            "Critical Indicator"
        ] = (
            data[
                "Customer Priority"
            ]
            .eq(
                "Critical"
            )
            .astype(int)
        )

        aggregation = {

            CUSTOMER_ID:
                "nunique",

            "High Risk Indicator":
                "sum",

            "Critical Indicator":
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

        result.columns = [
            (
                " ".join(
                    [
                        str(x)
                        for x in column
                        if str(x) != ""
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
            for column in result.columns
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

                "Critical Indicator sum":
                    "Critical Customers",

                "Churn Probability mean":
                    "Average Churn Probability",

                "Predicted 90-Day Revenue mean":
                    "Average Predicted Revenue",

                "Predicted 90-Day Revenue sum":
                    "Total Predicted Revenue"
            }
        )

        if (
            "Customers"
            in result.columns
        ):

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

        numeric = (
            result
            .select_dtypes(
                include="number"
            )
            .columns
        )

        result[
            numeric
        ] = result[
            numeric
        ].round(
            2
        )

        return (
            result
            .sort_values(
                by=(
                    "Total Predicted Revenue"
                    if
                    "Total Predicted Revenue"
                    in result.columns
                    else
                    "Customers"
                ),
                ascending=False
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

        data = (
            self.add_customer_priority(
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

        revenue = pd.to_numeric(
            data[
                "Predicted 90-Day Revenue"
            ],
            errors="coerce"
        ).fillna(
            0
        )

        data[
            "_Revenue"
        ] = revenue

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
                            "_Revenue",
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
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

    # =========================================================
    # PRIORITY CUSTOMERS
    # =========================================================

    def priority_customers(
        self,
        df: pd.DataFrame,
        top_n: int = 50
    ) -> pd.DataFrame:

        data = (
            self.add_customer_priority(
                df
            )
        )

        score_map = {

            "Critical":
                4,

            "High":
                3,

            "Medium":
                2,

            "Low":
                1,

            "Unknown":
                0
        }

        data[
            "_Priority Score"
        ] = (
            data[
                "Customer Priority"
            ]
            .map(
                score_map
            )
            .fillna(
                0
            )
        )

        if (
            "Predicted 90-Day Revenue"
            in data.columns
        ):

            data[
                "_Revenue"
            ] = (
                pd.to_numeric(
                    data[
                        "Predicted 90-Day Revenue"
                    ],
                    errors="coerce"
                )
                .fillna(
                    0
                )
            )

        else:

            data[
                "_Revenue"
            ] = 0

        result = (
            data
            .sort_values(
                by=[
                    "_Priority Score",
                    "_Revenue"
                ],
                ascending=[
                    False,
                    False
                ]
            )
            .head(
                top_n
            )
            .copy()
        )

        desired_columns = [
            CUSTOMER_ID,
            "Customer Segment",
            "Churn Probability",
            "Churn Risk",
            "Predicted 90-Day Revenue",
            "CLV Value Band",
            "Customer Priority",
            "Top Recommended Product",
            "Recommended Business Action"
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
    # COLOR KPI AREA
    # =========================================================

    def _render_kpis(
        self,
        metrics: dict
    ):

        row1 = (
            st.columns(
                4
            )
        )

        with row1[0]:

            render_color_card(
                title=
                    "Total Customers",

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
                    "High Risk Customers",

                value=
                    f"{metrics['high_risk_customers']:,}",

                icon=
                    "⚠️",

                card_class=
                    "card-red"
            )

        with row1[2]:

            render_color_card(
                title=
                    "High Value Customers",

                value=
                    f"{metrics['high_value_customers']:,}",

                icon=
                    "💎",

                card_class=
                    "card-purple"
            )

        with row1[3]:

            render_color_card(
                title=
                    "Critical Customers",

                value=
                    f"{metrics['critical_customers']:,}",

                icon=
                    "🔥",

                card_class=
                    "card-red"
            )

        st.write("")

        row2 = (
            st.columns(
                4
            )
        )

        with row2[0]:

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

        with row2[1]:

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

        with row2[2]:

            render_color_card(
                title=
                    "Average Churn Probability",

                value=
                    (
                        f"{metrics['average_churn_probability'] * 100:.1f}%"
                    ),

                icon=
                    "📊",

                card_class=
                    "card-purple"
            )

        with row2[3]:

            render_color_card(
                title=
                    "High Priority Customers",

                value=
                    (
                        f"{metrics['high_priority_customers']:,}"
                    ),

                icon=
                    "🎯",

                card_class=
                    "card-blue"
            )

    # =========================================================
    # REVENUE AT RISK GAUGE
    # =========================================================

    def _revenue_risk_gauge(
        self,
        metrics: dict
    ):

        value = float(
            metrics[
                "revenue_at_risk_percentage"
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
                        "%",
                    "font": {
                        "size":
                            38
                    }
                },

                title={
                    "text":
                        "Revenue at Risk"
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
                            "#ef4444"
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
                                60
                            ],
                            "color":
                                "#fef3c7"
                        },

                        {
                            "range": [
                                60,
                                100
                            ],
                            "color":
                                "#fee2e2"
                        }
                    ],

                    "threshold": {

                        "line": {
                            "color":
                                "#991b1b",

                            "width":
                                4
                        },

                        "thickness":
                            0.75,

                        "value":
                            value
                    }
                }
            )
        )

        figure.update_layout(
            height=330,
            margin=dict(
                l=20,
                r=20,
                t=70,
                b=20
            )
        )

        return figure

    # =========================================================
    # PRIORITY DONUT
    # =========================================================

    def _priority_donut(
        self,
        df: pd.DataFrame
    ):

        priority = (
            self.priority_distribution(
                df
            )
        )

        priority = (
            priority[
                priority[
                    "Customers"
                ]
                >
                0
            ]
        )

        figure = (
            px.pie(
                priority,
                names=
                    "Customer Priority",
                values=
                    "Customers",
                hole=
                    0.58,
                title=
                    "Customer Priority Mix",
                color=
                    "Customer Priority",
                color_discrete_map={
                    "Critical":
                        "#dc2626",

                    "High":
                        "#f97316",

                    "Medium":
                        "#eab308",

                    "Low":
                        "#10b981",

                    "Unknown":
                        "#94a3b8"
                }
            )
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
    # CHURN RISK DONUT
    # =========================================================

    def _churn_risk_donut(
        self,
        df: pd.DataFrame
    ):

        if (
            "Churn Risk"
            not in df.columns
        ):

            return None

        distribution = (
            df[
                "Churn Risk"
            ]
            .fillna(
                "Unknown"
            )
            .value_counts()
            .rename_axis(
                "Churn Risk"
            )
            .reset_index(
                name="Customers"
            )
        )

        figure = (
            px.pie(
                distribution,
                names=
                    "Churn Risk",
                values=
                    "Customers",
                hole=
                    0.58,
                title=
                    "Churn Risk Distribution",
                color=
                    "Churn Risk",
                color_discrete_map={
                    "High":
                        "#ef4444",

                    "Medium":
                        "#f59e0b",

                    "Low":
                        "#10b981",

                    "Unknown":
                        "#94a3b8"
                }
            )
        )

        figure.update_layout(
            height=380,
            legend_title_text=""
        )

        return figure

    # =========================================================
    # CLV DONUT
    # =========================================================

    def _clv_donut(
        self,
        df: pd.DataFrame
    ):

        if (
            "CLV Value Band"
            not in df.columns
        ):

            return None

        distribution = (
            df[
                "CLV Value Band"
            ]
            .fillna(
                "Unknown"
            )
            .value_counts()
            .rename_axis(
                "CLV Value Band"
            )
            .reset_index(
                name="Customers"
            )
        )

        figure = (
            px.pie(
                distribution,
                names=
                    "CLV Value Band",
                values=
                    "Customers",
                hole=
                    0.58,
                title=
                    "Customer Value Mix",
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
                }
            )
        )

        figure.update_layout(
            height=380,
            legend_title_text=""
        )

        return figure

    # =========================================================
    # VALUE RISK QUADRANT
    # =========================================================

    def _value_risk_scatter(
        self,
        df: pd.DataFrame
    ):

        data = (
            self.add_customer_priority(
                df
            )
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

        data[
            "Predicted 90-Day Revenue"
        ] = pd.to_numeric(
            data[
                "Predicted 90-Day Revenue"
            ],
            errors="coerce"
        )

        data[
            "Churn Probability"
        ] = pd.to_numeric(
            data[
                "Churn Probability"
            ],
            errors="coerce"
        )

        data = (
            data
            .dropna(
                subset=[
                    "Predicted 90-Day Revenue",
                    "Churn Probability"
                ]
            )
        )

        if data.empty:

            return None

        revenue_boundary = float(
            data[
                "Predicted 90-Day Revenue"
            ]
            .median()
        )

        churn_boundary = 0.70

        hover_columns = [
            CUSTOMER_ID
        ]

        for column in [
            "Customer Segment",
            "Churn Risk",
            "CLV Value Band",
            "Top Recommended Product"
        ]:

            if column in data.columns:

                hover_columns.append(
                    column
                )

        figure = (
            px.scatter(
                data,
                x=
                    "Predicted 90-Day Revenue",
                y=
                    "Churn Probability",
                color=
                    "Customer Priority",
                hover_data=
                    hover_columns,
                color_discrete_map={
                    "Critical":
                        "#dc2626",

                    "High":
                        "#f97316",

                    "Medium":
                        "#eab308",

                    "Low":
                        "#10b981",

                    "Unknown":
                        "#94a3b8"
                },
                title=
                    "Customer Value × Churn Risk Matrix"
            )
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
                "#64748b"
        )

        figure.add_annotation(
            x=
                revenue_boundary,
            y=
                0.98,
            text=
                "Higher Value →",
            showarrow=
                False,
            xshift=
                60
        )

        figure.add_annotation(
            x=
                revenue_boundary,
            y=
                churn_boundary,
            text=
                "High-value + High-risk zone",
            showarrow=
                False,
            xshift=
                115,
            yshift=
                35
        )

        figure.update_layout(
            height=520,

            xaxis_title=
                "Predicted 90-Day Revenue",

            yaxis_title=
                "Churn Probability",

            legend_title_text=
                "Customer Priority"
        )

        return figure

    # =========================================================
    # PRIORITY TREEMAP
    # =========================================================

    def _priority_treemap(
        self,
        df: pd.DataFrame
    ):

        data = (
            self.add_customer_priority(
                df
            )
        )

        if (
            "Customer Segment"
            not in data.columns
        ):

            return None

        grouped = (
            data
            .groupby(
                [
                    "Customer Segment",
                    "Customer Priority"
                ]
            )
            .agg(
                Customers=(
                    CUSTOMER_ID,
                    "nunique"
                )
            )
            .reset_index()
        )

        figure = (
            px.treemap(
                grouped,
                path=[
                    "Customer Segment",
                    "Customer Priority"
                ],
                values=
                    "Customers",
                color=
                    "Customer Priority",
                color_discrete_map={
                    "Critical":
                        "#dc2626",

                    "High":
                        "#f97316",

                    "Medium":
                        "#eab308",

                    "Low":
                        "#10b981",

                    "Unknown":
                        "#94a3b8"
                },
                title=
                    "Segment × Priority Treemap"
            )
        )

        figure.update_layout(
            height=480
        )

        return figure

    # =========================================================
    # STYLED PRIORITY TABLE
    # =========================================================

    def _styled_priority_table(
        self,
        df: pd.DataFrame
    ):

        table = (
            self.priority_customers(
                df,
                top_n=50
            )
        )

        if table.empty:

            st.info(
                "No priority customer data available."
            )

            return

        display = (
            table.copy()
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
            "Customer Priority"
            in display.columns
        ):

            display[
                "Customer Priority"
            ] = display[
                "Customer Priority"
            ].apply(
                format_priority_badge
            )

        if (
            "Churn Probability"
            in display.columns
        ):

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

        if (
            "Predicted 90-Day Revenue"
            in display.columns
        ):

            display[
                "Predicted 90-Day Revenue"
            ] = (
                pd.to_numeric(
                    display[
                        "Predicted 90-Day Revenue"
                    ],
                    errors="coerce"
                )
                .round(
                    2
                )
            )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            height=470
        )

    # =========================================================
    # BUSINESS INSIGHTS
    # =========================================================

    def _render_business_insights(
        self,
        metrics: dict,
        segment_table: pd.DataFrame
    ):

        st.subheader(
            "💡 Business Insights"
        )

        col1, col2, col3 = (
            st.columns(
                3
            )
        )

        with col1:

            st.info(
                (
                    f"⚠️ **{metrics['high_risk_customers']:,} "
                    f"customers** are currently classified "
                    f"as high churn risk."
                )
            )

        with col2:

            st.warning(
                (
                    f"📉 **{metrics['revenue_at_risk']:,.0f}** "
                    f"of predicted 90-day customer value "
                    f"is associated with high-risk customers."
                )
            )

        with col3:

            st.success(
                (
                    f"💎 **{metrics['high_value_customers']:,} "
                    f"customers** belong to the high-value "
                    f"CLV band."
                )
            )

        if (
            not segment_table.empty
            and
            "Total Predicted Revenue"
            in segment_table.columns
        ):

            top_segment = (
                segment_table
                .iloc[0]
            )

            st.success(
                (
                    f"🏆 **{top_segment['Customer Segment']}** "
                    f"is currently the largest predicted-revenue "
                    f"segment in the selected population."
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
            self.add_customer_priority(
                df
            )
        )

        metrics = (
            self.calculate_metrics(
                data
            )
        )

        # =====================================================
        # PAGE TITLE
        # =====================================================

        st.markdown(
            "## 🏠 Executive Intelligence Overview"
        )

        st.caption(
            "Unified business view of customer value, "
            "churn risk, priority, segments and predicted revenue."
        )

        # =====================================================
        # COLORED KPI CARDS
        # =====================================================

        self._render_kpis(
            metrics
        )

        st.write("")
        st.divider()

        # =====================================================
        # TABS
        # =====================================================

        tab1, tab2, tab3, tab4 = (
            st.tabs(
                [
                    "📊 Overview",
                    "🎯 Value × Risk",
                    "🧩 Segments",
                    "🔥 Priority Customers"
                ]
            )
        )

        # =====================================================
        # TAB 1 — OVERVIEW
        # =====================================================

        with tab1:

            col1, col2 = (
                st.columns(
                    2
                )
            )

            with col1:

                priority_figure = (
                    self._priority_donut(
                        data
                    )
                )

                st.plotly_chart(
                    priority_figure,
                    use_container_width=True
                )

            with col2:

                gauge = (
                    self._revenue_risk_gauge(
                        metrics
                    )
                )

                st.plotly_chart(
                    gauge,
                    use_container_width=True
                )

            col3, col4 = (
                st.columns(
                    2
                )
            )

            with col3:

                churn_donut = (
                    self._churn_risk_donut(
                        data
                    )
                )

                if churn_donut is not None:

                    st.plotly_chart(
                        churn_donut,
                        use_container_width=True
                    )

            with col4:

                clv_donut = (
                    self._clv_donut(
                        data
                    )
                )

                if clv_donut is not None:

                    st.plotly_chart(
                        clv_donut,
                        use_container_width=True
                    )

            segment_table = (
                self.segment_performance(
                    data
                )
            )

            self._render_business_insights(
                metrics,
                segment_table
            )

        # =====================================================
        # TAB 2 — VALUE × RISK
        # =====================================================

        with tab2:

            st.markdown(
                "### 🎯 Customer Value × Churn Risk"
            )

            st.caption(
                "Hover over any customer to inspect "
                "their segment, value band, churn risk "
                "and recommendation."
            )

            scatter = (
                self._value_risk_scatter(
                    data
                )
            )

            if scatter is None:

                st.info(
                    "Churn probability and predicted revenue "
                    "are required for this chart."
                )

            else:

                st.plotly_chart(
                    scatter,
                    use_container_width=True
                )

            st.info(
                "The upper-right area represents customers "
                "with relatively high predicted value and "
                "high churn probability. This is a useful "
                "retention-priority zone."
            )

        # =====================================================
        # TAB 3 — SEGMENTS
        # =====================================================

        with tab3:

            col1, col2 = (
                st.columns(
                    2
                )
            )

            segment_table = (
                self.segment_performance(
                    data
                )
            )

            with col1:

                if (
                    not segment_table.empty
                    and
                    "Total Predicted Revenue"
                    in segment_table.columns
                ):

                    segment_revenue_chart = (
                        px.bar(
                            segment_table,
                            x=
                                "Customer Segment",
                            y=
                                "Total Predicted Revenue",
                            color=
                                "Total Predicted Revenue",
                            text=
                                "Total Predicted Revenue",
                            title=
                                "Predicted Revenue by Segment",
                            color_continuous_scale=
                                "Viridis"
                        )
                    )

                    segment_revenue_chart.update_layout(
                        coloraxis_showscale=False
                    )

                    st.plotly_chart(
                        segment_revenue_chart,
                        use_container_width=True
                    )

            with col2:

                risk_segment = (
                    self.revenue_at_risk_by_segment(
                        data
                    )
                )

                if not risk_segment.empty:

                    risk_chart = (
                        px.bar(
                            risk_segment,
                            x=
                                "Customer Segment",
                            y=
                                "Revenue at Risk",
                            color=
                                "Revenue at Risk",
                            text=
                                "Revenue at Risk",
                            title=
                                "Revenue at Risk by Segment",
                            color_continuous_scale=
                                "Reds"
                        )
                    )

                    risk_chart.update_layout(
                        coloraxis_showscale=False
                    )

                    st.plotly_chart(
                        risk_chart,
                        use_container_width=True
                    )

            treemap = (
                self._priority_treemap(
                    data
                )
            )

            if treemap is not None:

                st.plotly_chart(
                    treemap,
                    use_container_width=True
                )

            st.markdown(
                "### 📋 Segment Performance"
            )

            if segment_table.empty:

                st.info(
                    "Segment performance data "
                    "is unavailable."
                )

            else:

                st.dataframe(
                    segment_table,
                    use_container_width=True,
                    hide_index=True
                )

        # =====================================================
        # TAB 4 — PRIORITY CUSTOMERS
        # =====================================================

        with tab4:

            st.markdown(
                "### 🔥 Priority Customer Action List"
            )

            st.caption(
                "Customers are ordered first by business "
                "priority and then by predicted 90-day revenue."
            )

            self._styled_priority_table(
                data
            )

            st.warning(
                "Customer Priority is a rule-based "
                "decision-support classification. "
                "It is not a separate ML prediction."
            )