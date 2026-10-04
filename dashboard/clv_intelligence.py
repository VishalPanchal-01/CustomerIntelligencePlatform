import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


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
        # Behaviour features
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

        # -----------------------------------------------------
        # CLV Value Band
        # -----------------------------------------------------

        if (
            "CLV Value Band"
            in data.columns
        ):

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

        return data

    # =========================================================
    # CALCULATE METRICS
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

        median_predicted_revenue = float(
            revenue.median()
        )

        maximum_predicted_revenue = float(
            revenue.max()
        )

        # -----------------------------------------------------
        # Value band metrics
        # -----------------------------------------------------

        if (
            "CLV Value Band"
            in data.columns
        ):

            value_band = (
                data[
                    "CLV Value Band"
                ]
                .astype(str)
                .str.lower()
            )

            high_value_customers = int(
                (
                    value_band
                    ==
                    "high"
                )
                .sum()
            )

            medium_value_customers = int(
                (
                    value_band
                    ==
                    "medium"
                )
                .sum()
            )

            low_value_customers = int(
                (
                    value_band
                    ==
                    "low"
                )
                .sum()
            )

        else:

            high_value_customers = 0
            medium_value_customers = 0
            low_value_customers = 0

        # -----------------------------------------------------
        # High value share
        # -----------------------------------------------------

        if total_customers > 0:

            high_value_percentage = (
                high_value_customers
                /
                total_customers
                *
                100
            )

        else:

            high_value_percentage = 0.0

        # -----------------------------------------------------
        # Revenue from high-value customers
        # -----------------------------------------------------

        if (
            "CLV Value Band"
            in data.columns
        ):

            high_value_mask = (
                data[
                    "CLV Value Band"
                ]
                .astype(str)
                .str.lower()
                .eq(
                    "high"
                )
            )

            high_value_revenue = float(
                revenue[
                    high_value_mask
                ].sum()
            )

        else:

            high_value_revenue = 0.0

        # -----------------------------------------------------
        # High-value churn risk
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

            "total_predicted_revenue":
                total_predicted_revenue,

            "average_predicted_revenue":
                average_predicted_revenue,

            "median_predicted_revenue":
                median_predicted_revenue,

            "maximum_predicted_revenue":
                maximum_predicted_revenue,

            "high_value_customers":
                high_value_customers,

            "medium_value_customers":
                medium_value_customers,

            "low_value_customers":
                low_value_customers,

            "high_value_percentage":
                float(
                    high_value_percentage
                ),

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

        data = (
            self.prepare_data(
                df
            )
        )

        if (
            "CLV Value Band"
            not in data.columns
        ):

            return pd.DataFrame(
                columns=[
                    "CLV Value Band",
                    "Customers"
                ]
            )

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

        data = (
            self.prepare_data(
                df
            )
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
                ascending=
                    False
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

        data = (
            self.prepare_data(
                df
            )
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

        if (
            "CLV Value Band"
            in data.columns
        ):

            high_value_counts = (
                data[
                    data[
                        "CLV Value Band"
                    ]
                    .astype(str)
                    .str.lower()
                    .eq(
                        "high"
                    )
                ]
                .groupby(
                    "Customer Segment"
                )[
                    CUSTOMER_ID
                ]
                .nunique()
                .rename(
                    "High Value Customers"
                )
            )

            result = result.merge(
                high_value_counts,
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
                .fillna(
                    0
                )
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

    def high_value_at_risk(
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

        result = (
            data[
                mask
            ]
            .copy()
        )

        sort_columns = []

        ascending = []

        if (
            "Predicted 90-Day Revenue"
            in result.columns
        ):

            sort_columns.append(
                "Predicted 90-Day Revenue"
            )

            ascending.append(
                False
            )

        if (
            "Churn Probability"
            in result.columns
        ):

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
    # VALUE RANKING
    # =========================================================

    def customer_value_ranking(
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
            "Predicted 90-Day Revenue"
            not in data.columns
        ):

            return pd.DataFrame()

        result = (
            data
            .sort_values(
                by=
                    "Predicted 90-Day Revenue",
                ascending=
                    False
            )
            .head(
                top_n
            )
            .copy()
        )

        result[
            "Customer Value Rank"
        ] = (
            np.arange(
                1,
                len(result) + 1
            )
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
                ascending=
                    False
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
    # TOP CUSTOMER REVENUE SHARE
    # =========================================================

    def top_customer_revenue_share(
        self,
        df: pd.DataFrame,
        top_percentage: float = 20.0
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
                    len(
                        concentration
                    )
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
            "Customer Lifetime Value Intelligence"
        )

        st.caption(
            "Analyze predicted 90-day customer value, "
            "revenue concentration, customer segments "
            "and churn exposure."
        )

        if (
            "Predicted 90-Day Revenue"
            not in data.columns
        ):

            st.warning(
                "Predicted 90-Day Revenue data "
                "is not available."
            )

            return

        # =====================================================
        # METRICS
        # =====================================================

        metrics = (
            self.calculate_metrics(
                data
            )
        )

        top_20_share = (
            self.top_customer_revenue_share(
                data,
                top_percentage=20
            )
        )

        col1, col2, col3, col4 = (
            st.columns(
                4
            )
        )

        with col1:

            st.metric(
                "Total Predicted 90-Day Revenue",
                (
                    f"{metrics['total_predicted_revenue']:,.2f}"
                )
            )

        with col2:

            st.metric(
                "Average Predicted Revenue",
                (
                    f"{metrics['average_predicted_revenue']:,.2f}"
                )
            )

        with col3:

            st.metric(
                "High Value Customers",
                (
                    f"{metrics['high_value_customers']:,}"
                )
            )

        with col4:

            st.metric(
                "High-Value Customers at Risk",
                (
                    f"{metrics['high_value_at_risk']:,}"
                )
            )

        col5, col6, col7 = (
            st.columns(
                3
            )
        )

        with col5:

            st.metric(
                "Median Predicted Revenue",
                (
                    f"{metrics['median_predicted_revenue']:,.2f}"
                )
            )

        with col6:

            st.metric(
                "High Value Customer %",
                (
                    f"{metrics['high_value_percentage']:.2f}%"
                )
            )

        with col7:

            st.metric(
                "Revenue Share from Top 20% Customers",
                (
                    f"{top_20_share:.2f}%"
                )
            )

        st.divider()

        # =====================================================
        # REVENUE DISTRIBUTION
        # =====================================================

        left, right = (
            st.columns(
                2
            )
        )

        with left:

            st.subheader(
                "Predicted Revenue Distribution"
            )

            revenue_data = (
                data[
                    data[
                        "Predicted 90-Day Revenue"
                    ]
                    .notna()
                ]
            )

            histogram = (
                px.histogram(
                    revenue_data,
                    x=
                        "Predicted 90-Day Revenue",
                    nbins=
                        40,
                    title=
                        "Predicted 90-Day Revenue Distribution"
                )
            )

            histogram.update_layout(
                xaxis_title=
                    "Predicted 90-Day Revenue",

                yaxis_title=
                    "Customers"
            )

            st.plotly_chart(
                histogram,
                use_container_width=True
            )

        # =====================================================
        # VALUE BAND DISTRIBUTION
        # =====================================================

        with right:

            st.subheader(
                "CLV Value Band Distribution"
            )

            band_distribution = (
                self.value_band_distribution(
                    data
                )
            )

            if band_distribution.empty:

                st.info(
                    "CLV Value Band data "
                    "is not available."
                )

            else:

                band_chart = (
                    px.bar(
                        band_distribution,
                        x=
                            "CLV Value Band",
                        y=
                            "Customers",
                        text=
                            "Customers",
                        title=
                            "Customers by CLV Value Band"
                    )
                )

                st.plotly_chart(
                    band_chart,
                    use_container_width=True
                )

        st.divider()

        # =====================================================
        # REVENUE BY VALUE BAND
        # =====================================================

        st.subheader(
            "Revenue by CLV Value Band"
        )

        band_revenue = (
            self.revenue_by_value_band(
                data
            )
        )

        if band_revenue.empty:

            st.info(
                "CLV Value Band data "
                "is not available."
            )

        else:

            col1, col2 = (
                st.columns(
                    2
                )
            )

            with col1:

                chart = (
                    px.bar(
                        band_revenue,
                        x=
                            "CLV Value Band",
                        y=
                            "Total Predicted Revenue",
                        text=
                            "Total Predicted Revenue",
                        title=
                            "Predicted Revenue by Value Band"
                    )
                )

                st.plotly_chart(
                    chart,
                    use_container_width=True
                )

            with col2:

                pie = (
                    px.pie(
                        band_revenue,
                        names=
                            "CLV Value Band",
                        values=
                            "Total Predicted Revenue",
                        title=
                            "Revenue Contribution by Value Band"
                    )
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

        st.divider()

        # =====================================================
        # CLV BY SEGMENT
        # =====================================================

        st.subheader(
            "Customer Value by Segment"
        )

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

            col1, col2 = (
                st.columns(
                    2
                )
            )

            with col1:

                segment_revenue_chart = (
                    px.bar(
                        segment_clv,
                        x=
                            "Customer Segment",
                        y=
                            "Average Predicted Revenue",
                        text=
                            "Average Predicted Revenue",
                        title=
                            "Average Predicted Revenue by Segment"
                    )
                )

                st.plotly_chart(
                    segment_revenue_chart,
                    use_container_width=True
                )

            with col2:

                segment_total_chart = (
                    px.bar(
                        segment_clv,
                        x=
                            "Customer Segment",
                        y=
                            "Total Predicted Revenue",
                        text=
                            "Total Predicted Revenue",
                        title=
                            "Total Predicted Revenue by Segment"
                    )
                )

                st.plotly_chart(
                    segment_total_chart,
                    use_container_width=True
                )

            st.dataframe(
                segment_clv,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        # =====================================================
        # CLV VS CHURN
        # =====================================================

        st.subheader(
            "Customer Value vs Churn Probability"
        )

        if (
            "Churn Probability"
            in data.columns
        ):

            scatter_data = (
                data
                .dropna(
                    subset=[
                        "Predicted 90-Day Revenue",
                        "Churn Probability"
                    ]
                )
                .copy()
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

            color_column = (
                "CLV Value Band"
                if "CLV Value Band"
                in scatter_data.columns
                else None
            )

            scatter = (
                px.scatter(
                    scatter_data,
                    x=
                        "Predicted 90-Day Revenue",
                    y=
                        "Churn Probability",
                    color=
                        color_column,
                    hover_data=
                        hover_columns,
                    title=
                        (
                            "Predicted 90-Day Revenue "
                            "vs Churn Probability"
                        )
                )
            )

            st.plotly_chart(
                scatter,
                use_container_width=True
            )

        else:

            st.info(
                "Churn Probability data "
                "is not available."
            )

        st.divider()

        # =====================================================
        # REVENUE CONCENTRATION
        # =====================================================

        st.subheader(
            "Revenue Concentration"
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
                px.line(
                    concentration,
                    x=
                        "Customer Percentile",
                    y=
                        "Cumulative Revenue %",
                    title=
                        (
                            "Cumulative Predicted Revenue "
                            "by Customer Percentile"
                        )
                )
            )

            concentration_chart.update_layout(
                xaxis_title=
                    "Top Customers Included (%)",

                yaxis_title=
                    "Cumulative Predicted Revenue (%)"
            )

            st.plotly_chart(
                concentration_chart,
                use_container_width=True
            )

            st.caption(
                "Customers are sorted from highest to "
                "lowest predicted 90-day revenue. "
                "This chart shows how concentrated "
                "predicted revenue is among the "
                "highest-value customers."
            )

        st.divider()

        # =====================================================
        # HIGH VALUE AT RISK
        # =====================================================

        st.subheader(
            "High-Value Customers at Churn Risk"
        )

        high_value_risk = (
            self.high_value_at_risk(
                data,
                top_n=50
            )
        )

        if high_value_risk.empty:

            st.info(
                "No High CLV + High Churn Risk "
                "customers found for the selected filters."
            )

        else:

            st.dataframe(
                high_value_risk,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        # =====================================================
        # VALUE RANKING
        # =====================================================

        st.subheader(
            "Customer Value Ranking"
        )

        value_ranking = (
            self.customer_value_ranking(
                data,
                top_n=100
            )
        )

        st.dataframe(
            value_ranking,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Predicted 90-Day Revenue is the model's "
            "fixed-horizon future revenue estimate. "
            "It should not be interpreted as literal "
            "customer lifetime revenue."
        )