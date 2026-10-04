import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


CUSTOMER_ID = "Customer ID"


class ExecutiveOverview:

    # =========================================================
    # PRIORITY CLASSIFICATION
    # =========================================================

    def add_customer_priority(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        """
        Creates business-oriented customer priority labels
        by combining churn risk and CLV value band.

        Priority logic:

        Critical
            High CLV + High Churn Risk

        High
            High CLV + Medium Churn Risk
            Medium CLV + High Churn Risk

        Medium
            High CLV + Low Churn Risk
            Medium CLV + Medium Churn Risk
            Low CLV + High Churn Risk

        Low
            Remaining combinations
        """

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
    # CLASSIFY PRIORITY
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

        # -----------------------------------------------------
        # Critical
        # -----------------------------------------------------

        if (
            churn == "high"
            and
            value == "high"
        ):

            return "Critical"

        # -----------------------------------------------------
        # High
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Medium
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Unknown
        # -----------------------------------------------------

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
                "Immediate retention attention; "
                "protect high-value customer relationship."
            )

        if priority == "High":

            return (
                "Prioritize retention campaign and "
                "personalized engagement."
            )

        if priority == "Medium":

            return (
                "Monitor customer behaviour and use "
                "targeted recommendations."
            )

        if priority == "Low":

            return (
                "Maintain regular engagement and "
                "continue personalized recommendations."
            )

        return (
            "Insufficient intelligence for "
            "priority assignment."
        )

    # =========================================================
    # EXECUTIVE METRICS
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
        # Revenue
        # -----------------------------------------------------

        if (
            "Predicted 90-Day Revenue"
            in data.columns
        ):

            revenue = pd.to_numeric(
                data[
                    "Predicted 90-Day Revenue"
                ],
                errors="coerce"
            ).fillna(
                0
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
        # Revenue at risk
        # -----------------------------------------------------

        high_risk_mask = (
            data[
                "Churn Risk"
            ]
            .astype(str)
            .str.lower()
            ==
            "high"
        )

        revenue_at_risk = float(
            revenue[
                high_risk_mask
            ].sum()
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
        # Critical customers
        # -----------------------------------------------------

        critical_customers = int(
            (
                data[
                    "Customer Priority"
                ]
                ==
                "Critical"
            )
            .sum()
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

        # -----------------------------------------------------
        # High-value high-risk revenue
        # -----------------------------------------------------

        critical_mask = (
            data[
                "Customer Priority"
            ]
            ==
            "Critical"
        )

        critical_revenue = float(
            revenue[
                critical_mask
            ].sum()
        )

        # -----------------------------------------------------
        # Churn probability
        # -----------------------------------------------------

        if (
            "Churn Probability"
            in data.columns
        ):

            churn_probability = pd.to_numeric(
                data[
                    "Churn Probability"
                ],
                errors="coerce"
            )

            average_churn_probability = float(
                churn_probability.mean()
            )

        else:

            average_churn_probability = 0.0

        return {

            "total_customers":
                total_customers,

            "total_predicted_revenue":
                total_predicted_revenue,

            "average_predicted_revenue":
                average_predicted_revenue,

            "revenue_at_risk":
                revenue_at_risk,

            "revenue_at_risk_percentage":
                float(
                    revenue_at_risk_percentage
                ),

            "critical_customers":
                critical_customers,

            "high_priority_customers":
                high_priority_customers,

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

        priority_order = [
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
                priority_order,
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

        # -----------------------------------------------------
        # Numeric preparation
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

            "High Value Indicator":
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

        rename_mapping = {

            f"{CUSTOMER_ID} nunique":
                "Customers",

            "High Risk Indicator sum":
                "High Risk Customers",

            "High Value Indicator sum":
                "High Value Customers",

            "Critical Indicator sum":
                "Critical Customers",

            "Churn Probability mean":
                "Average Churn Probability",

            "Predicted 90-Day Revenue mean":
                "Average Predicted Revenue",

            "Predicted 90-Day Revenue sum":
                "Total Predicted Revenue"
        }

        result = result.rename(
            columns=
                rename_mapping
        )

        # -----------------------------------------------------
        # Percent calculations
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

        # -----------------------------------------------------
        # Formatting
        # -----------------------------------------------------

        numeric_columns = result.select_dtypes(
            include="number"
        ).columns

        result[
            numeric_columns
        ] = result[
            numeric_columns
        ].round(
            2
        )

        if (
            "Total Predicted Revenue"
            in result.columns
        ):

            result = result.sort_values(
                by=
                    "Total Predicted Revenue",

                ascending=
                    False
            )

        return (
            result
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

        priority_score = {

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
                priority_score
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
            ] = pd.to_numeric(
                data[
                    "Predicted 90-Day Revenue"
                ],
                errors="coerce"
            ).fillna(
                0
            )

        else:

            data[
                "_Revenue"
            ] = 0

        data = (
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
            if column in data.columns
        ]

        return (
            data[
                available_columns
            ]
            .reset_index(
                drop=True
            )
        )

    # =========================================================
    # RENDER EXECUTIVE OVERVIEW
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
        # HEADER
        # =====================================================

        st.title(
            "Executive Customer Intelligence"
        )

        st.caption(
            "Business-oriented overview combining "
            "customer churn risk, predicted value, "
            "segmentation and product recommendations."
        )

        # =====================================================
        # PRIMARY KPIs
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
                "90-Day Revenue at Risk",
                (
                    f"{metrics['revenue_at_risk']:,.2f}"
                )
            )

        with col3:

            st.metric(
                "Critical Customers",
                f"{metrics['critical_customers']:,}"
            )

        with col4:

            st.metric(
                "Critical Customer Revenue",
                (
                    f"{metrics['critical_customer_revenue']:,.2f}"
                )
            )

        # =====================================================
        # SECONDARY KPIs
        # =====================================================

        col5, col6, col7, col8 = (
            st.columns(
                4
            )
        )

        with col5:

            st.metric(
                "Total Predicted 90-Day Revenue",
                (
                    f"{metrics['total_predicted_revenue']:,.2f}"
                )
            )

        with col6:

            st.metric(
                "Revenue at Risk %",
                (
                    f"{metrics['revenue_at_risk_percentage']:.2f}%"
                )
            )

        with col7:

            st.metric(
                "High Priority Customers",
                f"{metrics['high_priority_customers']:,}"
            )

        with col8:

            st.metric(
                "Average Churn Probability",
                (
                    f"{metrics['average_churn_probability'] * 100:.2f}%"
                )
            )

        st.divider()

        # =====================================================
        # PRIORITY + VALUE/RISK MATRIX
        # =====================================================

        col1, col2 = (
            st.columns(
                2
            )
        )

        # -----------------------------------------------------
        # Priority distribution
        # -----------------------------------------------------

        with col1:

            st.subheader(
                "Customer Priority Distribution"
            )

            priority_df = (
                self.priority_distribution(
                    data
                )
            )

            priority_chart = (
                px.bar(
                    priority_df,
                    x=
                        "Customer Priority",
                    y=
                        "Customers",
                    text=
                        "Customers",
                    title=
                        "Customers by Business Priority"
                )
            )

            priority_chart.update_layout(
                xaxis_title=
                    "Priority",

                yaxis_title=
                    "Customers"
            )

            st.plotly_chart(
                priority_chart,
                use_container_width=True
            )

        # -----------------------------------------------------
        # Churn × value matrix
        # -----------------------------------------------------

        with col2:

            st.subheader(
                "Customer Value vs Churn Risk"
            )

            if (
                "Predicted 90-Day Revenue"
                in data.columns
                and
                "Churn Probability"
                in data.columns
            ):

                scatter_data = (
                    data.copy()
                )

                scatter_data[
                    "Predicted 90-Day Revenue"
                ] = pd.to_numeric(
                    scatter_data[
                        "Predicted 90-Day Revenue"
                    ],
                    errors="coerce"
                )

                scatter_data[
                    "Churn Probability"
                ] = pd.to_numeric(
                    scatter_data[
                        "Churn Probability"
                    ],
                    errors="coerce"
                )

                scatter_data = (
                    scatter_data
                    .dropna(
                        subset=[
                            "Predicted 90-Day Revenue",
                            "Churn Probability"
                        ]
                    )
                )

                hover_columns = [
                    CUSTOMER_ID
                ]

                for column in [
                    "Customer Segment",
                    "CLV Value Band",
                    "Churn Risk",
                    "Top Recommended Product"
                ]:

                    if column in scatter_data.columns:

                        hover_columns.append(
                            column
                        )

                scatter_chart = (
                    px.scatter(
                        scatter_data,
                        x=
                            "Predicted 90-Day Revenue",
                        y=
                            "Churn Probability",
                        color=
                            "Customer Priority",
                        hover_data=
                            hover_columns,
                        title=
                            (
                                "Predicted Customer Value "
                                "vs Churn Probability"
                            )
                    )
                )

                scatter_chart.update_layout(
                    xaxis_title=
                        "Predicted 90-Day Revenue",

                    yaxis_title=
                        "Churn Probability"
                )

                st.plotly_chart(
                    scatter_chart,
                    use_container_width=True
                )

            else:

                st.info(
                    "Churn and CLV data are required "
                    "for the customer value-risk matrix."
                )

        st.divider()

        # =====================================================
        # CHURN RISK × CLV MATRIX
        # =====================================================

        st.subheader(
            "Risk-Value Customer Matrix"
        )

        if (
            "Churn Risk"
            in data.columns
            and
            "CLV Value Band"
            in data.columns
        ):

            matrix = pd.crosstab(
                data[
                    "CLV Value Band"
                ],
                data[
                    "Churn Risk"
                ]
            )

            matrix = (
                matrix
                .reset_index()
            )

            st.dataframe(
                matrix,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "CLV Value Band and Churn Risk "
                "are required for this matrix."
            )

        st.divider()

        # =====================================================
        # SEGMENT PERFORMANCE
        # =====================================================

        st.subheader(
            "Segment Performance"
        )

        segment_table = (
            self.segment_performance(
                data
            )
        )

        if segment_table.empty:

            st.info(
                "Customer Segment data is not available."
            )

        else:

            st.dataframe(
                segment_table,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        # =====================================================
        # PRIORITY CUSTOMERS
        # =====================================================

        st.subheader(
            "Priority Customer Action List"
        )

        priority_table = (
            self.priority_customers(
                data,
                top_n=50
            )
        )

        st.dataframe(
            priority_table,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Priority classification is a dashboard "
            "decision-support rule combining the existing "
            "CLV value band and churn-risk outputs. "
            "It is not a separate machine-learning model."
        )