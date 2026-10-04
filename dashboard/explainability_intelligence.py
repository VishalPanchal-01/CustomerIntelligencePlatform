import json
import os

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


class ExplainabilityIntelligence:

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        project_root: str
    ):

        self.project_root = (
            project_root
        )

        self.churn_directory = os.path.join(
            project_root,
            "artifacts",
            "explainability",
            "churn"
        )

        self.clv_directory = os.path.join(
            project_root,
            "artifacts",
            "explainability",
            "clv"
        )

        self.churn_importance_path = os.path.join(
            self.churn_directory,
            "churn_global_shap_importance.csv"
        )

        self.clv_importance_path = os.path.join(
            self.clv_directory,
            "clv_global_shap_importance.csv"
        )

        self.churn_shap_path = os.path.join(
            self.churn_directory,
            "churn_dashboard_shap_values.csv"
        )

        self.clv_shap_path = os.path.join(
            self.clv_directory,
            "clv_dashboard_shap_values.csv"
        )

        self.churn_summary_path = os.path.join(
            self.churn_directory,
            "churn_visualization_summary.json"
        )

        self.clv_summary_path = os.path.join(
            self.clv_directory,
            "clv_visualization_summary.json"
        )


    # =========================================================
    # CUSTOMER ID NORMALIZATION
    # =========================================================

    @staticmethod
    def normalize_customer_id(
        value
    ) -> str:

        if pd.isna(
            value
        ):

            return ""

        text = str(
            value
        ).strip()

        # Example:
        # 17850.0 -> 17850
        try:

            numeric = float(
                text
            )

            if numeric.is_integer():

                return str(
                    int(
                        numeric
                    )
                )

        except (
            TypeError,
            ValueError
        ):

            pass

        return text


    # =========================================================
    # NORMALIZE CUSTOMER ID COLUMN
    # =========================================================

    def _normalize_customer_column(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = df.copy()

        if (
            "CustomerID" in data.columns
            and
            CUSTOMER_ID not in data.columns
        ):

            data = data.rename(
                columns={
                    "CustomerID":
                        CUSTOMER_ID
                }
            )

        if CUSTOMER_ID in data.columns:

            data[CUSTOMER_ID] = (
                data[CUSTOMER_ID]
                .apply(
                    self.normalize_customer_id
                )
            )

        return data


    # =========================================================
    # LOAD CSV
    # =========================================================

    def _load_csv(
        self,
        path: str
    ) -> pd.DataFrame:

        if not os.path.exists(
            path
        ):

            return pd.DataFrame()

        data = pd.read_csv(
            path
        )

        return (
            self._normalize_customer_column(
                data
            )
        )


    # =========================================================
    # LOAD JSON
    # =========================================================

    @staticmethod
    def _load_json(
        path: str
    ) -> dict:

        if not os.path.exists(
            path
        ):

            return {}

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(
                file
            )


    # =========================================================
    # LOAD ARTIFACTS
    # =========================================================

    def load_artifacts(
        self
    ) -> dict:

        return {

            "churn_importance":
                self._load_csv(
                    self.churn_importance_path
                ),

            "clv_importance":
                self._load_csv(
                    self.clv_importance_path
                ),

            "churn_shap":
                self._load_csv(
                    self.churn_shap_path
                ),

            "clv_shap":
                self._load_csv(
                    self.clv_shap_path
                ),

            "churn_summary":
                self._load_json(
                    self.churn_summary_path
                ),

            "clv_summary":
                self._load_json(
                    self.clv_summary_path
                )
        }


    # =========================================================
    # GLOBAL IMPORTANCE DATA
    # =========================================================

    @staticmethod
    def prepare_global_importance(
        importance_df: pd.DataFrame,
        model_name: str
    ) -> pd.DataFrame:

        if importance_df.empty:

            return pd.DataFrame()

        required = [
            "Feature",
            "Mean Absolute SHAP"
        ]

        if any(
            column not in importance_df.columns
            for column in required
        ):

            return pd.DataFrame()

        result = (
            importance_df[
                required
            ]
            .copy()
        )

        result[
            "Mean Absolute SHAP"
        ] = pd.to_numeric(
            result[
                "Mean Absolute SHAP"
            ],
            errors="coerce"
        )

        result = (
            result
            .dropna(
                subset=[
                    "Mean Absolute SHAP"
                ]
            )
        )

        result[
            "Model"
        ] = model_name

        return (
            result
            .sort_values(
                by=
                    "Mean Absolute SHAP",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )


    # =========================================================
    # NORMALIZED GLOBAL IMPORTANCE
    # =========================================================

    def normalized_importance_comparison(
        self,
        churn_importance: pd.DataFrame,
        clv_importance: pd.DataFrame
    ) -> pd.DataFrame:

        frames = []

        churn = (
            self.prepare_global_importance(
                churn_importance,
                "Churn"
            )
        )

        clv = (
            self.prepare_global_importance(
                clv_importance,
                "Predicted 90-Day Revenue"
            )
        )

        for frame in [
            churn,
            clv
        ]:

            if frame.empty:

                continue

            maximum = float(
                frame[
                    "Mean Absolute SHAP"
                ]
                .max()
            )

            frame = frame.copy()

            if maximum > 0:

                frame[
                    "Normalized Importance"
                ] = (
                    frame[
                        "Mean Absolute SHAP"
                    ]
                    /
                    maximum
                    *
                    100
                )

            else:

                frame[
                    "Normalized Importance"
                ] = 0.0

            frames.append(
                frame
            )

        if not frames:

            return pd.DataFrame()

        return pd.concat(
            frames,
            ignore_index=True
        )


    # =========================================================
    # AVAILABLE EXPLAINED CUSTOMERS
    # =========================================================

    def available_customers(
        self,
        churn_shap: pd.DataFrame,
        clv_shap: pd.DataFrame
    ) -> list:

        churn_customers = set()

        clv_customers = set()

        if (
            not churn_shap.empty
            and
            CUSTOMER_ID in churn_shap.columns
        ):

            churn_customers = set(
                churn_shap[
                    CUSTOMER_ID
                ]
                .dropna()
                .astype(str)
            )

        if (
            not clv_shap.empty
            and
            CUSTOMER_ID in clv_shap.columns
        ):

            clv_customers = set(
                clv_shap[
                    CUSTOMER_ID
                ]
                .dropna()
                .astype(str)
            )

        # Prefer customers available in both modules
        common = sorted(
            churn_customers
            &
            clv_customers
        )

        if common:

            return common

        return sorted(
            churn_customers
            |
            clv_customers
        )


    # =========================================================
    # CUSTOMER SHAP DATA
    # =========================================================

    def customer_shap_data(
        self,
        shap_df: pd.DataFrame,
        customer_id
    ) -> pd.DataFrame:

        if (
            shap_df.empty
            or
            CUSTOMER_ID not in shap_df.columns
        ):

            return pd.DataFrame()

        normalized_id = (
            self.normalize_customer_id(
                customer_id
            )
        )

        result = (
            shap_df[
                shap_df[
                    CUSTOMER_ID
                ]
                ==
                normalized_id
            ]
            .copy()
        )

        if result.empty:

            return result

        numeric_columns = [
            "Feature Value",
            "SHAP Value",
            "Absolute SHAP",
            "Impact Rank"
        ]

        for column in numeric_columns:

            if column in result.columns:

                result[column] = pd.to_numeric(
                    result[column],
                    errors="coerce"
                )

        if "Absolute SHAP" not in result.columns:

            if "SHAP Value" in result.columns:

                result[
                    "Absolute SHAP"
                ] = (
                    result[
                        "SHAP Value"
                    ]
                    .abs()
                )

        if "Absolute SHAP" in result.columns:

            result = (
                result
                .sort_values(
                    by=
                        "Absolute SHAP",
                    ascending=False
                )
            )

        return (
            result
            .reset_index(
                drop=True
            )
        )


    # =========================================================
    # TOP POSITIVE / NEGATIVE DRIVERS
    # =========================================================

    @staticmethod
    def driver_summary(
        explanation_df: pd.DataFrame,
        top_n: int = 3
    ) -> dict:

        if (
            explanation_df.empty
            or
            "SHAP Value"
            not in explanation_df.columns
        ):

            return {
                "positive":
                    pd.DataFrame(),

                "negative":
                    pd.DataFrame()
            }

        positive = (
            explanation_df[
                explanation_df[
                    "SHAP Value"
                ]
                >
                0
            ]
            .sort_values(
                by=
                    "Absolute SHAP",
                ascending=False
            )
            .head(
                top_n
            )
        )

        negative = (
            explanation_df[
                explanation_df[
                    "SHAP Value"
                ]
                <
                0
            ]
            .sort_values(
                by=
                    "Absolute SHAP",
                ascending=False
            )
            .head(
                top_n
            )
        )

        return {
            "positive":
                positive,

            "negative":
                negative
        }


    # =========================================================
    # CUSTOMER MASTER RECORD
    # =========================================================

    def find_customer_record(
        self,
        customer_df: pd.DataFrame,
        customer_id
    ):

        if (
            customer_df.empty
            or
            CUSTOMER_ID not in customer_df.columns
        ):

            return None

        data = customer_df.copy()

        normalized_id = (
            self.normalize_customer_id(
                customer_id
            )
        )

        ids = (
            data[
                CUSTOMER_ID
            ]
            .apply(
                self.normalize_customer_id
            )
        )

        result = (
            data[
                ids
                ==
                normalized_id
            ]
        )

        if result.empty:

            return None

        return result.iloc[0]


    # =========================================================
    # GLOBAL IMPORTANCE CHART
    # =========================================================

    @staticmethod
    def _importance_chart(
        data: pd.DataFrame,
        title: str
    ):

        if data.empty:

            return None

        plot_data = (
            data
            .sort_values(
                by=
                    "Mean Absolute SHAP",
                ascending=True
            )
        )

        figure = px.bar(
            plot_data,

            x=
                "Mean Absolute SHAP",

            y=
                "Feature",

            orientation=
                "h",

            color=
                "Mean Absolute SHAP",

            text=
                "Mean Absolute SHAP",

            color_continuous_scale=
                "Viridis",

            title=
                title
        )

        figure.update_traces(
            texttemplate=
                "%{text:.3f}"
        )

        figure.update_layout(
            coloraxis_showscale=False,
            height=420
        )

        return figure


    # =========================================================
    # DRIVER CHART
    # =========================================================

    @staticmethod
    def _driver_chart(
        explanation_df: pd.DataFrame,
        title: str,
        positive_label: str,
        negative_label: str
    ):

        if explanation_df.empty:

            return None

        plot_data = (
            explanation_df
            .sort_values(
                by=
                    "SHAP Value"
            )
            .copy()
        )

        plot_data[
            "Direction"
        ] = plot_data[
            "SHAP Value"
        ].apply(
            lambda value:
                positive_label
                if value > 0
                else negative_label
                if value < 0
                else "Neutral"
        )

        figure = px.bar(
            plot_data,

            x=
                "SHAP Value",

            y=
                "Feature",

            orientation=
                "h",

            color=
                "Direction",

            text=
                "SHAP Value",

            hover_data=[
                "Feature Value"
            ],

            color_discrete_map={
                positive_label:
                    "#ef4444",

                negative_label:
                    "#10b981",

                "Neutral":
                    "#94a3b8"
            },

            title=
                title
        )

        figure.update_traces(
            texttemplate=
                "%{text:.3f}"
        )

        figure.add_vline(
            x=0,
            line_width=1,
            line_color="#475569"
        )

        figure.update_layout(
            legend_title_text="",
            height=450
        )

        return figure


    # =========================================================
    # INTERPRETATION TABLE
    # =========================================================

    @staticmethod
    def _render_driver_tables(
        explanation_df: pd.DataFrame,
        positive_title: str,
        negative_title: str
    ):

        summary = (
            ExplainabilityIntelligence
            .driver_summary(
                explanation_df,
                top_n=3
            )
        )

        col1, col2 = st.columns(
            2
        )

        with col1:

            st.markdown(
                f"#### ⬆️ {positive_title}"
            )

            positive = (
                summary[
                    "positive"
                ]
            )

            if positive.empty:

                st.info(
                    "No positive SHAP drivers "
                    "in the saved explanation."
                )

            else:

                st.dataframe(
                    positive[
                        [
                            "Feature",
                            "Feature Value",
                            "SHAP Value"
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

        with col2:

            st.markdown(
                f"#### ⬇️ {negative_title}"
            )

            negative = (
                summary[
                    "negative"
                ]
            )

            if negative.empty:

                st.info(
                    "No negative SHAP drivers "
                    "in the saved explanation."
                )

            else:

                st.dataframe(
                    negative[
                        [
                            "Feature",
                            "Feature Value",
                            "SHAP Value"
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True
                )


    # =========================================================
    # MODEL STATUS CARDS
    # =========================================================

    @staticmethod
    def _render_model_status_cards(
        churn_importance: pd.DataFrame,
        clv_importance: pd.DataFrame,
        churn_shap: pd.DataFrame,
        clv_shap: pd.DataFrame
    ):

        churn_customers = (
            churn_shap[
                CUSTOMER_ID
            ]
            .nunique()
            if
            not churn_shap.empty
            and
            CUSTOMER_ID in churn_shap.columns
            else
            0
        )

        clv_customers = (
            clv_shap[
                CUSTOMER_ID
            ]
            .nunique()
            if
            not clv_shap.empty
            and
            CUSTOMER_ID in clv_shap.columns
            else
            0
        )

        churn_top = (
            churn_importance.iloc[0][
                "Feature"
            ]
            if
            not churn_importance.empty
            else
            "N/A"
        )

        clv_top = (
            clv_importance.iloc[0][
                "Feature"
            ]
            if
            not clv_importance.empty
            else
            "N/A"
        )

        cols = st.columns(
            4
        )

        with cols[0]:

            render_color_card(
                title=
                    "Top Churn Driver",

                value=
                    churn_top,

                icon=
                    "⚠️",

                card_class=
                    "card-red"
            )

        with cols[1]:

            render_color_card(
                title=
                    "Top Value Driver",

                value=
                    clv_top,

                icon=
                    "💰",

                card_class=
                    "card-green"
            )

        with cols[2]:

            render_color_card(
                title=
                    "Churn Customers Explained",

                value=
                    f"{churn_customers:,}",

                icon=
                    "🧠",

                card_class=
                    "card-purple"
            )

        with cols[3]:

            render_color_card(
                title=
                    "CLV Customers Explained",

                value=
                    f"{clv_customers:,}",

                icon=
                    "🔍",

                card_class=
                    "card-blue"
            )


    # =========================================================
    # RENDER
    # =========================================================

    def render(
        self,
        customer_df: pd.DataFrame
    ):

        artifacts = (
            self.load_artifacts()
        )

        churn_importance = (
            self.prepare_global_importance(
                artifacts[
                    "churn_importance"
                ],
                "Churn"
            )
        )

        clv_importance = (
            self.prepare_global_importance(
                artifacts[
                    "clv_importance"
                ],
                "Predicted 90-Day Revenue"
            )
        )

        churn_shap = (
            artifacts[
                "churn_shap"
            ]
        )

        clv_shap = (
            artifacts[
                "clv_shap"
            ]
        )

        # =====================================================
        # PAGE HEADER
        # =====================================================

        st.markdown(
            "## 🧠 Explainability Intelligence"
        )

        st.caption(
            "Understand why the churn and predicted "
            "90-day revenue models produce their outputs "
            "using SHAP feature contributions."
        )

        # =====================================================
        # ARTIFACT CHECK
        # =====================================================

        if (
            churn_importance.empty
            and
            clv_importance.empty
        ):

            st.error(
                "SHAP artifacts have not been generated yet."
            )

            st.info(
                "Run:\n\n"
                "`python generate_churn_shap_visualizations.py`\n\n"
                "and\n\n"
                "`python generate_clv_shap_visualizations.py`"
            )

            return

        # =====================================================
        # KPI CARDS
        # =====================================================

        self._render_model_status_cards(
            churn_importance,
            clv_importance,
            churn_shap,
            clv_shap
        )

        st.write("")
        st.divider()

        # =====================================================
        # TABS
        # =====================================================

        tab1, tab2, tab3 = (
            st.tabs(
                [
                    "🌍 Global Explainability",
                    "👤 Customer Explainability",
                    "📘 How to Interpret SHAP"
                ]
            )
        )

        # =====================================================
        # TAB 1 — GLOBAL
        # =====================================================

        with tab1:

            st.markdown(
                "### 🌍 Global Model Explainability"
            )

            col1, col2 = st.columns(
                2
            )

            with col1:

                churn_chart = (
                    self._importance_chart(
                        churn_importance,
                        (
                            "Churn Model — "
                            "Global SHAP Importance"
                        )
                    )
                )

                if churn_chart is not None:

                    st.plotly_chart(
                        churn_chart,
                        use_container_width=True
                    )

                else:

                    st.info(
                        "Churn SHAP importance "
                        "is not available."
                    )

            with col2:

                clv_chart = (
                    self._importance_chart(
                        clv_importance,
                        (
                            "Predicted 90-Day Revenue — "
                            "Global SHAP Importance"
                        )
                    )
                )

                if clv_chart is not None:

                    st.plotly_chart(
                        clv_chart,
                        use_container_width=True
                    )

                else:

                    st.info(
                        "CLV SHAP importance "
                        "is not available."
                    )

            st.markdown(
                "### 🔄 Normalized Importance Comparison"
            )

            comparison = (
                self.normalized_importance_comparison(
                    artifacts[
                        "churn_importance"
                    ],
                    artifacts[
                        "clv_importance"
                    ]
                )
            )

            if comparison.empty:

                st.info(
                    "Importance comparison "
                    "is not available."
                )

            else:

                comparison_chart = px.bar(
                    comparison,

                    x=
                        "Feature",

                    y=
                        "Normalized Importance",

                    color=
                        "Model",

                    barmode=
                        "group",

                    text=
                        "Normalized Importance",

                    title=
                        (
                            "Relative Feature Importance "
                            "Within Each Model"
                        )
                )

                comparison_chart.update_traces(
                    texttemplate=
                        "%{text:.1f}"
                )

                comparison_chart.update_layout(
                    yaxis_title=
                        "Normalized Importance (0–100)",

                    legend_title_text=
                        ""
                )

                st.plotly_chart(
                    comparison_chart,
                    use_container_width=True
                )

            st.info(
                "The normalized comparison is relative "
                "within each model. Raw SHAP magnitudes "
                "from churn and revenue models should not "
                "be directly compared because their model "
                "outputs are on different scales."
            )

        # =====================================================
        # TAB 2 — CUSTOMER
        # =====================================================

        with tab2:

            st.markdown(
                "### 👤 Customer-Level Explanation"
            )

            customers = (
                self.available_customers(
                    churn_shap,
                    clv_shap
                )
            )

            if not customers:

                st.warning(
                    "No customer-level SHAP "
                    "explanations are available."
                )

            else:

                selected_customer = (
                    st.selectbox(
                        "Select Customer ID",

                        options=
                            customers,

                        key=
                            "shap_customer_selector"
                    )
                )

                customer_record = (
                    self.find_customer_record(
                        customer_df,
                        selected_customer
                    )
                )

                churn_customer = (
                    self.customer_shap_data(
                        churn_shap,
                        selected_customer
                    )
                )

                clv_customer = (
                    self.customer_shap_data(
                        clv_shap,
                        selected_customer
                    )
                )

                # -------------------------------------------------
                # CUSTOMER SUMMARY CARDS
                # -------------------------------------------------

                if customer_record is not None:

                    cards = st.columns(
                        4
                    )

                    churn_probability = (
                        customer_record.get(
                            "Churn Probability"
                        )
                    )

                    if pd.notna(
                        churn_probability
                    ):

                        churn_probability_text = (
                            f"{float(churn_probability) * 100:.1f}%"
                        )

                    else:

                        churn_probability_text = "N/A"

                    predicted_revenue = (
                        customer_record.get(
                            "Predicted 90-Day Revenue"
                        )
                    )

                    if pd.notna(
                        predicted_revenue
                    ):

                        revenue_text = (
                            f"{float(predicted_revenue):,.0f}"
                        )

                    else:

                        revenue_text = "N/A"

                    with cards[0]:

                        render_color_card(
                            title=
                                "Churn Probability",

                            value=
                                churn_probability_text,

                            icon=
                                "⚠️",

                            card_class=
                                "card-red"
                        )

                    with cards[1]:

                        render_color_card(
                            title=
                                "Churn Risk",

                            value=
                                format_risk_badge(
                                    customer_record.get(
                                        "Churn Risk",
                                        "Unknown"
                                    )
                                ),

                            icon=
                                "📉",

                            card_class=
                                "card-red"
                        )

                    with cards[2]:

                        render_color_card(
                            title=
                                "Predicted 90-Day Revenue",

                            value=
                                revenue_text,

                            icon=
                                "💰",

                            card_class=
                                "card-green"
                        )

                    with cards[3]:

                        render_color_card(
                            title=
                                "CLV Value Band",

                            value=
                                format_clv_badge(
                                    customer_record.get(
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

                # -------------------------------------------------
                # CHURN EXPLANATION
                # -------------------------------------------------

                st.markdown(
                    "### ⚠️ Why did the churn model "
                    "make this prediction?"
                )

                if churn_customer.empty:

                    st.info(
                        "This customer is not present "
                        "in the saved churn SHAP sample."
                    )

                else:

                    churn_chart = (
                        self._driver_chart(
                            churn_customer,

                            (
                                "Customer Churn "
                                "SHAP Contributions"
                            ),

                            "Increases Churn",

                            "Decreases Churn"
                        )
                    )

                    st.plotly_chart(
                        churn_chart,
                        use_container_width=True
                    )

                    self._render_driver_tables(
                        churn_customer,

                        "Top Churn Drivers",

                        "Top Retention Drivers"
                    )

                st.divider()

                # -------------------------------------------------
                # CLV EXPLANATION
                # -------------------------------------------------

                st.markdown(
                    "### 💰 Why did the revenue model "
                    "make this prediction?"
                )

                if clv_customer.empty:

                    st.info(
                        "This customer is not present "
                        "in the saved CLV SHAP sample."
                    )

                else:

                    clv_chart = (
                        self._driver_chart(
                            clv_customer,

                            (
                                "Predicted 90-Day Revenue "
                                "SHAP Contributions"
                            ),

                            "Increases Revenue",

                            "Decreases Revenue"
                        )
                    )

                    st.plotly_chart(
                        clv_chart,
                        use_container_width=True
                    )

                    self._render_driver_tables(
                        clv_customer,

                        "Top Value-Increasing Factors",

                        "Top Value-Decreasing Factors"
                    )

                st.warning(
                    "SHAP explains how features influenced "
                    "the model output relative to its baseline. "
                    "It does not prove that those features "
                    "caused the customer's future behavior."
                )

        # =====================================================
        # TAB 3 — INTERPRETATION
        # =====================================================

        with tab3:

            st.markdown(
                "### 📘 How to Read SHAP"
            )

            st.markdown(
                """
**For the churn model**

- Positive SHAP → pushes the churn prediction upward.
- Negative SHAP → pushes the churn prediction downward.
- Large absolute SHAP → stronger model contribution.

**For the predicted 90-day revenue model**

- Positive SHAP → pushes predicted revenue upward.
- Negative SHAP → pushes predicted revenue downward.
- Large absolute SHAP → stronger contribution to the model output.

**Global importance**

Mean absolute SHAP tells us which features generally
have the strongest influence across the explained customers.

**Local explanation**

Customer-level SHAP explains why one specific customer
received their model prediction.
                """
            )

            st.info(
                "Churn SHAP values and CLV SHAP values "
                "should not be compared directly in raw "
                "magnitude because the two model outputs "
                "operate on different scales."
            )

            st.warning(
                "SHAP is an explainability technique, "
                "not a causal-inference method."
            )