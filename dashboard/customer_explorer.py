import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.executive_overview import ExecutiveOverview
from dashboard.churn_intelligence import ChurnIntelligence
from dashboard.recommendation_intelligence import RecommendationIntelligence

from dashboard.ui import (
    render_color_card,
    format_risk_badge,
    format_clv_badge,
    format_priority_badge
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
    # NORMALIZE CUSTOMER ID
    # =========================================================

    @staticmethod
    def normalize_customer_id(value) -> str:

        if value is None:
            return ""

        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass

        text = str(value).strip()

        if not text:
            return ""

        try:
            numeric_value = float(text)

            if (
                np.isfinite(numeric_value)
                and
                numeric_value.is_integer()
            ):
                return str(int(numeric_value))

        except (
            TypeError,
            ValueError,
            OverflowError
        ):
            pass

        return text

    # =========================================================
    # PREPARE DATA
    # =========================================================

    def prepare_data(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        if df is None:
            raise ValueError(
                "Customer dataframe cannot be None."
            )

        data = df.copy()

        # -----------------------------------------------------
        # Customer ID backward compatibility
        # -----------------------------------------------------

        if (
            CUSTOMER_ID not in data.columns
            and
            "CustomerID" in data.columns
        ):
            data = data.rename(
                columns={
                    "CustomerID": CUSTOMER_ID
                }
            )

        if (
            CUSTOMER_ID not in data.columns
            and
            "Customer Id" in data.columns
        ):
            data = data.rename(
                columns={
                    "Customer Id": CUSTOMER_ID
                }
            )

        if CUSTOMER_ID not in data.columns:
            raise ValueError(
                "Customer ID column not found."
            )

        # -----------------------------------------------------
        # Keep Customer ID original datatype
        # -----------------------------------------------------

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

                data[column] = pd.to_numeric(
                    data[column],
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

        data = self.prepare_data(
            df
        )

        normalized_customer_id = (
            self.normalize_customer_id(
                customer_id
            )
        )

        normalized_ids = (
            data[CUSTOMER_ID]
            .apply(
                self.normalize_customer_id
            )
        )

        result = data[
            normalized_ids
            ==
            normalized_customer_id
        ]

        if result.empty:
            return None

        return result.iloc[0]

    # =========================================================
    # SHAP DRIVER SUMMARY
    # =========================================================

    @staticmethod
    def shap_driver_summary(
        explanation_df: pd.DataFrame,
        top_n: int = 3
    ) -> dict:

        empty_result = {
            "positive": pd.DataFrame(),
            "negative": pd.DataFrame()
        }

        if explanation_df is None:
            return empty_result

        if explanation_df.empty:
            return empty_result

        if "SHAP Value" not in explanation_df.columns:
            return empty_result

        data = explanation_df.copy()

        data["SHAP Value"] = pd.to_numeric(
            data["SHAP Value"],
            errors="coerce"
        )

        data = (
            data
            .dropna(
                subset=[
                    "SHAP Value"
                ]
            )
            .copy()
        )

        if data.empty:
            return empty_result

        if "Absolute SHAP" not in data.columns:

            data["Absolute SHAP"] = (
                data["SHAP Value"].abs()
            )

        else:

            data["Absolute SHAP"] = pd.to_numeric(
                data["Absolute SHAP"],
                errors="coerce"
            )

            missing_mask = (
                data["Absolute SHAP"].isna()
            )

            data.loc[
                missing_mask,
                "Absolute SHAP"
            ] = (
                data.loc[
                    missing_mask,
                    "SHAP Value"
                ]
                .abs()
            )

        positive = (
            data[
                data["SHAP Value"] > 0
            ]
            .sort_values(
                "Absolute SHAP",
                ascending=False
            )
            .head(top_n)
            .reset_index(drop=True)
        )

        negative = (
            data[
                data["SHAP Value"] < 0
            ]
            .sort_values(
                "Absolute SHAP",
                ascending=False
            )
            .head(top_n)
            .reset_index(drop=True)
        )

        return {
            "positive": positive,
            "negative": negative
        }

    # =========================================================
    # BUSINESS PRIORITY
    # =========================================================

    def customer_priority(
        self,
        df: pd.DataFrame,
        customer_id
    ) -> dict:

        data = self.prepare_data(
            df
        )

        customer = self.get_customer(
            data,
            customer_id
        )

        if customer is None:

            return {
                "priority": "Unknown",
                "business_action": "Customer not found."
            }

        # -----------------------------------------------------
        # IMPORTANT
        #
        # We already found the correct customer.
        # Do not search/filter using the ID again.
        # -----------------------------------------------------

        customer_df = (
            customer
            .to_frame()
            .T
        )

        overview = ExecutiveOverview()

        result_df = (
            overview
            .add_customer_priority(
                customer_df
            )
        )

        if result_df.empty:

            return {
                "priority": "Unknown",
                "business_action": "Customer priority unavailable."
            }

        result = result_df.iloc[0]

        return {
            "priority": result.get(
                "Customer Priority",
                "Unknown"
            ),

            "business_action": result.get(
                "Recommended Business Action",
                "No action available."
            )
        }

    # =========================================================
    # RETENTION PRIORITY SCORE
    # =========================================================

    def retention_priority_score(
        self,
        df: pd.DataFrame,
        customer_id
    ):

        data = self.prepare_data(
            df
        )

        customer = self.get_customer(
            data,
            customer_id
        )

        if customer is None:
            return None

        actual_customer_id = customer.get(
            CUSTOMER_ID
        )

        churn = ChurnIntelligence()

        ranking = (
            churn
            .retention_priority_customers(
                data,
                top_n=len(data)
            )
        )

        if ranking.empty:
            return None

        if CUSTOMER_ID not in ranking.columns:
            return None

        normalized_target = (
            self.normalize_customer_id(
                actual_customer_id
            )
        )

        normalized_ids = (
            ranking[CUSTOMER_ID]
            .apply(
                self.normalize_customer_id
            )
        )

        result = ranking[
            normalized_ids
            ==
            normalized_target
        ]

        if result.empty:
            return None

        value = result[
            "Retention Priority Score"
        ].iloc[0]

        if pd.isna(value):
            return None

        return float(value)

    # =========================================================
    # RETENTION RANK
    # =========================================================

    def retention_rank(
        self,
        df: pd.DataFrame,
        customer_id
    ):

        data = self.prepare_data(
            df
        )

        customer = self.get_customer(
            data,
            customer_id
        )

        if customer is None:
            return None

        actual_customer_id = customer.get(
            CUSTOMER_ID
        )

        churn = ChurnIntelligence()

        ranking = (
            churn
            .retention_priority_customers(
                data,
                top_n=len(data)
            )
        )

        if ranking.empty:
            return None

        if CUSTOMER_ID not in ranking.columns:
            return None

        ranking = (
            ranking
            .copy()
            .reset_index(
                drop=True
            )
        )

        ranking[
            "Retention Rank"
        ] = np.arange(
            1,
            len(ranking) + 1
        )

        normalized_target = (
            self.normalize_customer_id(
                actual_customer_id
            )
        )

        normalized_ids = (
            ranking[CUSTOMER_ID]
            .apply(
                self.normalize_customer_id
            )
        )

        result = ranking[
            normalized_ids
            ==
            normalized_target
        ]

        if result.empty:
            return None

        return int(
            result[
                "Retention Rank"
            ]
            .iloc[0]
        )

    # =========================================================
    # BEHAVIOR PERCENTILES
    # =========================================================

    def behavior_percentiles(
        self,
        df: pd.DataFrame,
        customer_id
    ) -> pd.DataFrame:

        data = self.prepare_data(
            df
        )

        customer = self.get_customer(
            data,
            customer_id
        )

        if customer is None:
            return pd.DataFrame()

        rows = []

        for feature in self.BEHAVIOR_FEATURES:

            if feature not in data.columns:
                continue

            population = (
                data[feature]
                .dropna()
            )

            value = customer.get(
                feature
            )

            if (
                population.empty
                or
                pd.isna(value)
            ):
                continue

            percentile = float(
                (
                    population
                    <=
                    value
                )
                .mean()
                *
                100
            )

            rows.append(
                {
                    "Feature": feature,
                    "Customer Value": float(value),
                    "Population Average": float(
                        population.mean()
                    ),
                    "Percentile": percentile
                }
            )

        return pd.DataFrame(
            rows
        )

    # =========================================================
    # NORMALIZED BEHAVIOR PROFILE
    # =========================================================

    def normalized_behavior_profile(
        self,
        df: pd.DataFrame,
        customer_id
    ) -> pd.DataFrame:

        profile = (
            self.behavior_percentiles(
                df,
                customer_id
            )
        )

        if profile.empty:
            return pd.DataFrame()

        profile = (
            profile[
                [
                    "Feature",
                    "Percentile"
                ]
            ]
            .copy()
        )

        recency_mask = (
            profile["Feature"]
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
    # RECOMMENDATIONS
    # =========================================================

    def recommendations(
        self,
        df: pd.DataFrame,
        customer_id
    ) -> pd.DataFrame:

        data = self.prepare_data(
            df
        )

        customer = self.get_customer(
            data,
            customer_id
        )

        if customer is None:
            return pd.DataFrame()

        # -----------------------------------------------------
        # We already have the exact customer row.
        #
        # Create a one-row DataFrame and send the exact
        # Customer ID contained in that row.
        # -----------------------------------------------------

        customer_df = (
            customer
            .to_frame()
            .T
        )

        actual_customer_id = (
            customer_df[
                CUSTOMER_ID
            ]
            .iloc[0]
        )

        recommender = (
            RecommendationIntelligence()
        )

        result = (
            recommender
            .customer_recommendations(
                customer_df,
                actual_customer_id
            )
        )

        if result is None:
            return pd.DataFrame()

        return result

    # =========================================================
    # CUSTOMER REPORT
    # =========================================================

    def build_customer_report(
        self,
        df: pd.DataFrame,
        customer_id
    ) -> pd.DataFrame:

        data = self.prepare_data(
            df
        )

        customer = self.get_customer(
            data,
            customer_id
        )

        if customer is None:
            return pd.DataFrame()

        actual_customer_id = (
            customer.get(
                CUSTOMER_ID
            )
        )

        priority = (
            self.customer_priority(
                data,
                actual_customer_id
            )
        )

        retention_score = (
            self.retention_priority_score(
                data,
                actual_customer_id
            )
        )

        retention_rank = (
            self.retention_rank(
                data,
                actual_customer_id
            )
        )

        rows = [
            {
                "Category": "Identity",
                "Metric": "Customer ID",
                "Value": customer.get(
                    CUSTOMER_ID
                )
            },
            {
                "Category": "Segmentation",
                "Metric": "Customer Segment",
                "Value": customer.get(
                    "Customer Segment",
                    "Not Available"
                )
            },
            {
                "Category": "Churn",
                "Metric": "Churn Probability",
                "Value": customer.get(
                    "Churn Probability",
                    "Not Available"
                )
            },
            {
                "Category": "Churn",
                "Metric": "Churn Risk",
                "Value": customer.get(
                    "Churn Risk",
                    "Not Available"
                )
            },
            {
                "Category": "CLV",
                "Metric": "Predicted 90-Day Revenue",
                "Value": customer.get(
                    "Predicted 90-Day Revenue",
                    "Not Available"
                )
            },
            {
                "Category": "CLV",
                "Metric": "CLV Value Band",
                "Value": customer.get(
                    "CLV Value Band",
                    "Not Available"
                )
            },
            {
                "Category": "Business Priority",
                "Metric": "Customer Priority",
                "Value": priority[
                    "priority"
                ]
            },
            {
                "Category": "Business Priority",
                "Metric": "Retention Priority Score",
                "Value": retention_score
            },
            {
                "Category": "Business Priority",
                "Metric": "Retention Rank",
                "Value": retention_rank
            },
            {
                "Category": "Recommendation",
                "Metric": "Top Recommended Product",
                "Value": customer.get(
                    "Top Recommended Product",
                    "Not Available"
                )
            },
            {
                "Category": "Business Action",
                "Metric": "Recommended Business Action",
                "Value": priority[
                    "business_action"
                ]
            }
        ]

        for feature in self.BEHAVIOR_FEATURES:

            if feature in customer.index:

                rows.append(
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
            rows
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
    # PROFILE HEADER
    # =========================================================

    def _render_profile_header(
        self,
        customer,
        priority: dict,
        retention_rank
    ):

        customer_id = customer.get(
            CUSTOMER_ID,
            ""
        )

        segment = customer.get(
            "Customer Segment",
            "Unknown"
        )

        priority_text = (
            format_priority_badge(
                priority[
                    "priority"
                ]
            )
        )

        rank_text = (
            f"Retention Rank #{retention_rank}"
            if retention_rank is not None
            else
            "Retention Rank N/A"
        )

        html = (
            '<div style="'
            'background:linear-gradient('
            '120deg,#1e1b4b,#4f46e5,#7c3aed);'
            'padding:28px 32px;'
            'border-radius:22px;'
            'color:white;'
            'box-shadow:0 14px 35px '
            'rgba(79,70,229,0.25);'
            'margin-bottom:24px;'
            '">'
            '<div style="font-size:0.9rem;'
            'opacity:0.8;">CUSTOMER 360 PROFILE</div>'
            f'<div style="font-size:2.2rem;'
            f'font-weight:800;'
            f'margin-top:4px;">'
            f'👤 Customer {customer_id}'
            f'</div>'
            f'<div style="font-size:1rem;'
            f'margin-top:8px;">'
            f'🧩 {segment}'
            f' &nbsp;&nbsp;•&nbsp;&nbsp; '
            f'{priority_text}'
            f' &nbsp;&nbsp;•&nbsp;&nbsp; '
            f'🎯 {rank_text}'
            f'</div>'
            '</div>'
        )

        st.markdown(
            html,
            unsafe_allow_html=True
        )

    # =========================================================
    # RETENTION GAUGE
    # =========================================================

    def _retention_gauge(
        self,
        score
    ):

        if score is None:
            value = 0.0
        else:
            value = float(score) * 100

        value = float(
            np.clip(
                value,
                0,
                100
            )
        )

        figure = go.Figure(
            go.Indicator(
                mode="gauge+number",

                value=value,

                number={
                    "suffix": "%"
                },

                title={
                    "text":
                        "Retention Priority Score"
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
                                40
                            ],
                            "color":
                                "#dcfce7"
                        },
                        {
                            "range": [
                                40,
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
    # BEHAVIOR RADAR
    # =========================================================

    def _behavior_radar(
        self,
        df: pd.DataFrame,
        customer_id
    ):

        profile = (
            self.normalized_behavior_profile(
                df,
                customer_id
            )
        )

        if profile.empty:
            return None

        categories = (
            profile["Feature"]
            .tolist()
        )

        values = (
            profile["Percentile"]
            .tolist()
        )

        if not categories:
            return None

        categories = (
            categories
            +
            [
                categories[0]
            ]
        )

        values = (
            values
            +
            [
                values[0]
            ]
        )

        figure = go.Figure()

        figure.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                name="Customer",

                line={
                    "color":
                        "#7c3aed",
                    "width":
                        3
                },

                fillcolor=
                    "rgba(124,58,237,0.20)"
            )
        )

        figure.update_layout(
            polar={
                "radialaxis": {
                    "visible": True,
                    "range": [
                        0,
                        100
                    ]
                }
            },

            title=
                "Customer Behavioral Position",

            height=500,

            showlegend=False
        )

        return figure

    # =========================================================
    # BEHAVIOR COMPARISON
    # =========================================================

    def _behavior_comparison_chart(
        self,
        df: pd.DataFrame,
        customer_id
    ):

        profile = (
            self.behavior_percentiles(
                df,
                customer_id
            )
        )

        if profile.empty:
            return None

        chart_data = (
            profile
            .melt(
                id_vars=[
                    "Feature"
                ],

                value_vars=[
                    "Customer Value",
                    "Population Average"
                ],

                var_name=
                    "Measure",

                value_name=
                    "Value"
            )
        )

        figure = px.bar(
            chart_data,

            x="Feature",
            y="Value",
            color="Measure",

            barmode="group",

            title=
                "Customer vs Population Average"
        )

        figure.update_layout(
            height=450
        )

        return figure

    # =========================================================
    # PRODUCT CARDS
    # =========================================================

    def _render_product_cards(
        self,
        recommendations: pd.DataFrame
    ):

        if (
            recommendations is None
            or
            recommendations.empty
        ):

            st.info(
                "No Top-N recommendations available."
            )

            return

        cards = recommendations.head(
            5
        )

        columns = st.columns(
            len(cards)
        )

        for position, (
            _,
            row
        ) in enumerate(
            cards.iterrows()
        ):

            with columns[position]:

                rank = row.get(
                    "Rank",
                    position + 1
                )

                product = row.get(
                    "Product",
                    "Unknown Product"
                )

                code = row.get(
                    "Stock Code",
                    ""
                )

                st.markdown(
                    (
                        '<div style="'
                        'background:linear-gradient('
                        '135deg,#2563eb,#7c3aed);'
                        'padding:18px;'
                        'border-radius:18px;'
                        'color:white;'
                        'min-height:190px;'
                        'box-shadow:0 10px 30px '
                        'rgba(79,70,229,0.22);'
                        '">'
                        f'<div style="font-size:0.8rem;'
                        f'opacity:0.8;">'
                        f'RECOMMENDATION #{rank}'
                        f'</div>'
                        f'<div style="font-size:1rem;'
                        f'font-weight:700;'
                        f'margin-top:12px;">'
                        f'{product}'
                        f'</div>'
                        f'<div style="font-size:0.8rem;'
                        f'margin-top:18px;'
                        f'opacity:0.85;">'
                        f'Stock Code: {code}'
                        f'</div>'
                        '</div>'
                    ),

                    unsafe_allow_html=True
                )

    # =========================================================
    # BUSINESS ACTION PANEL
    # =========================================================

    def _render_action_panel(
        self,
        priority: dict
    ):

        value = priority.get(
            "priority",
            "Unknown"
        )

        action = priority.get(
            "business_action",
            "No action available."
        )

        if value == "Critical":

            background = (
                "linear-gradient(135deg,#991b1b,#dc2626)"
            )

            icon = "🔥"

        elif value == "High":

            background = (
                "linear-gradient(135deg,#c2410c,#f97316)"
            )

            icon = "⚠️"

        elif value == "Medium":

            background = (
                "linear-gradient(135deg,#a16207,#eab308)"
            )

            icon = "🎯"

        else:

            background = (
                "linear-gradient(135deg,#047857,#10b981)"
            )

            icon = "✅"

        html = (
            f'<div style="'
            f'background:{background};'
            f'padding:24px;'
            f'border-radius:18px;'
            f'color:white;'
            f'box-shadow:0 10px 25px '
            f'rgba(0,0,0,0.12);'
            f'">'
            f'<div style="font-size:0.85rem;'
            f'opacity:0.85;">'
            f'RECOMMENDED BUSINESS ACTION'
            f'</div>'
            f'<div style="font-size:1.25rem;'
            f'font-weight:700;'
            f'margin-top:8px;">'
            f'{icon} {value} Priority'
            f'</div>'
            f'<div style="font-size:0.95rem;'
            f'margin-top:12px;'
            f'line-height:1.6;">'
            f'{action}'
            f'</div>'
            f'</div>'
        )

        st.markdown(
            html,
            unsafe_allow_html=True
        )

    # =========================================================
    # COMPLETE RECORD TABLE
    # =========================================================

    def _styled_record(
        self,
        customer
    ):

        record = (
            customer
            .to_frame(
                name="Value"
            )
            .reset_index()
            .rename(
                columns={
                    "index":
                        "Metric"
                }
            )
        )

        st.dataframe(
            record,
            use_container_width=True,
            hide_index=True,
            height=520
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

        st.markdown(
            "## 👤 360° Customer Explorer"
        )

        st.caption(
            "A unified CRM-style customer profile combining "
            "behavior, churn, predicted value, segmentation "
            "and product recommendations."
        )

        customer_options = (
            data[CUSTOMER_ID]
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        if not customer_options:

            st.warning(
                "No valid customer IDs are available."
            )

            return

        selected_customer = (
            st.selectbox(
                "Select Customer ID",

                options=
                    customer_options,

                key=
                    "premium_360_customer"
            )
        )

        customer = self.get_customer(
            data,
            selected_customer
        )

        if customer is None:

            st.warning(
                "Customer not found."
            )

            return

        actual_customer_id = (
            customer.get(
                CUSTOMER_ID
            )
        )

        priority = (
            self.customer_priority(
                data,
                actual_customer_id
            )
        )

        retention_score = (
            self.retention_priority_score(
                data,
                actual_customer_id
            )
        )

        retention_rank = (
            self.retention_rank(
                data,
                actual_customer_id
            )
        )

        # =====================================================
        # PROFILE HEADER
        # =====================================================

        self._render_profile_header(
            customer,
            priority,
            retention_rank
        )

        # =====================================================
        # TOP KPI CARDS
        # =====================================================

        row1 = st.columns(
            4
        )

        churn_probability = (
            customer.get(
                "Churn Probability"
            )
        )

        if pd.notna(
            churn_probability
        ):

            churn_value = (
                f"{float(churn_probability) * 100:.1f}%"
            )

        else:

            churn_value = "N/A"

        revenue = (
            customer.get(
                "Predicted 90-Day Revenue"
            )
        )

        if pd.notna(
            revenue
        ):

            revenue_value = (
                f"{float(revenue):,.0f}"
            )

        else:

            revenue_value = "N/A"

        with row1[0]:

            render_color_card(
                title=
                    "Churn Probability",

                value=
                    churn_value,

                icon=
                    "⚠️",

                card_class=
                    "card-red"
            )

        with row1[1]:

            render_color_card(
                title=
                    "Churn Risk",

                value=
                    format_risk_badge(
                        customer.get(
                            "Churn Risk",
                            "Unknown"
                        )
                    ),

                icon=
                    "📉",

                card_class=
                    "card-red"
            )

        with row1[2]:

            render_color_card(
                title=
                    "Predicted 90-Day Revenue",

                value=
                    revenue_value,

                icon=
                    "💰",

                card_class=
                    "card-green"
            )

        with row1[3]:

            render_color_card(
                title=
                    "CLV Value Band",

                value=
                    format_clv_badge(
                        customer.get(
                            "CLV Value Band",
                            "Unknown"
                        )
                    ),

                icon=
                    "💎",

                card_class=
                    "card-purple"
            )

        st.write("")

        row2 = st.columns(
            3
        )

        with row2[0]:

            render_color_card(
                title=
                    "Customer Segment",

                value=
                    customer.get(
                        "Customer Segment",
                        "N/A"
                    ),

                icon=
                    "🧩",

                card_class=
                    "card-blue"
            )

        with row2[1]:

            render_color_card(
                title=
                    "Customer Priority",

                value=
                    format_priority_badge(
                        priority[
                            "priority"
                        ]
                    ),

                icon=
                    "🔥",

                card_class=
                    "card-purple"
            )

        with row2[2]:

            render_color_card(
                title=
                    "Retention Rank",

                value=(
                    f"#{retention_rank}"
                    if retention_rank is not None
                    else
                    "N/A"
                ),

                icon=
                    "🎯",

                card_class=
                    "card-green"
            )

        st.write("")

        st.divider()

        # =====================================================
        # TABS
        # =====================================================

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📊 Customer Summary",
                "🧠 Behaviour Profile",
                "🎯 Recommendations",
                "📄 Report & Export"
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

                gauge = self._retention_gauge(
                    retention_score
                )

                st.plotly_chart(
                    gauge,
                    use_container_width=True
                )

            with col2:

                st.markdown(
                    "### 🎯 Recommended Action"
                )

                self._render_action_panel(
                    priority
                )

                st.info(
                    "Customer Priority and Retention Priority "
                    "are decision-support rules built from "
                    "existing ML outputs. They are not "
                    "additional trained models."
                )

            st.markdown(
                "### 📌 Customer Snapshot"
            )

            snapshot = []

            for field in [
                "Recency",
                "Frequency",
                "Monetary",
                "AverageOrderValue",
                "Tenure",
                "Top Recommended Product"
            ]:

                if field in customer.index:

                    snapshot.append(
                        {
                            "Metric":
                                field,

                            "Value":
                                customer.get(
                                    field
                                )
                        }
                    )

            st.dataframe(
                pd.DataFrame(
                    snapshot
                ),

                use_container_width=True,

                hide_index=True
            )

        # =====================================================
        # TAB 2
        # =====================================================

        with tab2:

            col1, col2 = st.columns(
                2
            )

            with col1:

                radar = (
                    self._behavior_radar(
                        data,
                        actual_customer_id
                    )
                )

                if radar is not None:

                    st.plotly_chart(
                        radar,
                        use_container_width=True
                    )

            with col2:

                comparison = (
                    self._behavior_comparison_chart(
                        data,
                        actual_customer_id
                    )
                )

                if comparison is not None:

                    st.plotly_chart(
                        comparison,
                        use_container_width=True
                    )

            percentiles = (
                self.behavior_percentiles(
                    data,
                    actual_customer_id
                )
            )

            if not percentiles.empty:

                st.markdown(
                    "### 📊 Population Percentiles"
                )

                percentile_chart = px.bar(
                    percentiles,

                    x=
                        "Feature",

                    y=
                        "Percentile",

                    color=
                        "Percentile",

                    text=
                        "Percentile",

                    color_continuous_scale=
                        "Viridis",

                    hover_data=[
                        "Customer Value",
                        "Population Average"
                    ],

                    title=
                        "Customer Position in Overall Population"
                )

                percentile_chart.update_layout(
                    coloraxis_showscale=False,

                    yaxis_range=[
                        0,
                        100
                    ]
                )

                st.plotly_chart(
                    percentile_chart,
                    use_container_width=True
                )

            st.caption(
                "For raw Recency, a lower value means "
                "the customer purchased more recently. "
                "The radar reverses Recency so a higher "
                "engagement score represents more recent activity."
            )

        # =====================================================
        # TAB 3
        # =====================================================

        with tab3:

            st.markdown(
                "### 🎯 Personalized Recommendations"
            )

            recommendations = (
                self.recommendations(
                    data,
                    actual_customer_id
                )
            )

            self._render_product_cards(
                recommendations
            )

            st.write("")

            if (
                "Recommendation Source"
                in customer.index
            ):

                st.info(
                    (
                        "Recommendation Source: "
                        f"**{customer.get('Recommendation Source', 'N/A')}**"
                    )
                )

            if (
                "Top Recommendation Score"
                in customer.index
                and
                pd.notna(
                    customer.get(
                        "Top Recommendation Score"
                    )
                )
            ):

                st.success(
                    (
                        "Top Recommendation Ranking Score: "
                        f"**{float(customer['Top Recommendation Score']):.4f}**"
                    )
                )

            if (
                recommendations is not None
                and
                not recommendations.empty
            ):

                st.dataframe(
                    recommendations,
                    use_container_width=True,
                    hide_index=True
                )

            st.warning(
                "Recommendation scores are ranking signals. "
                "They should not be interpreted as purchase "
                "probabilities."
            )

        # =====================================================
        # TAB 4
        # =====================================================

        with tab4:

            st.markdown(
                "### 📄 Complete Customer Intelligence Record"
            )

            self._styled_record(
                customer
            )

            st.divider()

            st.markdown(
                "### 📥 Export Customer Intelligence Report"
            )

            report = (
                self.build_customer_report(
                    data,
                    actual_customer_id
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
                    actual_customer_id
                )
            )

            safe_customer_id = (
                self.normalize_customer_id(
                    actual_customer_id
                )
                .replace(
                    " ",
                    "_"
                )
                .replace(
                    "/",
                    "_"
                )
                .replace(
                    "\\",
                    "_"
                )
            )

            st.download_button(
                label=
                    "📥 Download Customer Intelligence Report",

                data=
                    report_bytes,

                file_name=
                    (
                        f"customer_"
                        f"{safe_customer_id}_"
                        f"intelligence_report.csv"
                    ),

                mime=
                    "text/csv",

                use_container_width=
                    True
            )