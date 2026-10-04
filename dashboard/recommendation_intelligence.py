import pandas as pd
import plotly.express as px
import streamlit as st


CUSTOMER_ID = "Customer ID"


class RecommendationIntelligence:

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
        # Recommendation score
        # -----------------------------------------------------

        if (
            "Top Recommendation Score"
            in data.columns
        ):

            data[
                "Top Recommendation Score"
            ] = pd.to_numeric(
                data[
                    "Top Recommendation Score"
                ],
                errors="coerce"
            )

        # -----------------------------------------------------
        # Recommendation text fields
        # -----------------------------------------------------

        text_columns = [
            "Top Recommended Stock Code",
            "Top Recommended Product",
            "Recommendation Source",
            "Recommended Stock Codes",
            "Recommended Products"
        ]

        for column in text_columns:

            if column in data.columns:

                data[
                    column
                ] = data[
                    column
                ].where(
                    data[
                        column
                    ].notna(),
                    None
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
        # Customers with recommendations
        # -----------------------------------------------------

        if (
            "Top Recommended Product"
            in data.columns
        ):

            customers_with_recommendations = int(
                data.loc[
                    data[
                        "Top Recommended Product"
                    ]
                    .notna(),
                    CUSTOMER_ID
                ]
                .nunique()
            )

            unique_top_products = int(
                data[
                    "Top Recommended Product"
                ]
                .dropna()
                .nunique()
            )

        else:

            customers_with_recommendations = 0
            unique_top_products = 0

        # -----------------------------------------------------
        # Coverage
        # -----------------------------------------------------

        if total_customers > 0:

            recommendation_coverage = (
                customers_with_recommendations
                /
                total_customers
                *
                100
            )

        else:

            recommendation_coverage = 0.0

        customers_without_recommendations = (
            total_customers
            -
            customers_with_recommendations
        )

        # -----------------------------------------------------
        # Average recommendation score
        # -----------------------------------------------------

        if (
            "Top Recommendation Score"
            in data.columns
        ):

            average_score = float(
                data[
                    "Top Recommendation Score"
                ]
                .dropna()
                .mean()
            )

        else:

            average_score = 0.0

        # -----------------------------------------------------
        # Unique products across complete Top-N lists
        # -----------------------------------------------------

        unique_recommended_products = (
            self._count_unique_recommended_products(
                data
            )
        )

        # -----------------------------------------------------
        # Average Top-N list size
        # -----------------------------------------------------

        average_recommendations_per_customer = (
            self._average_recommendation_count(
                data
            )
        )

        return {

            "total_customers":
                total_customers,

            "customers_with_recommendations":
                customers_with_recommendations,

            "customers_without_recommendations":
                customers_without_recommendations,

            "recommendation_coverage":
                float(
                    recommendation_coverage
                ),

            "unique_top_products":
                unique_top_products,

            "unique_recommended_products":
                unique_recommended_products,

            "average_top_recommendation_score":
                average_score,

            "average_recommendations_per_customer":
                average_recommendations_per_customer
        }

    # =========================================================
    # UNIQUE PRODUCTS IN COMPLETE LISTS
    # =========================================================

    def _count_unique_recommended_products(
        self,
        df: pd.DataFrame
    ) -> int:

        if (
            "Recommended Products"
            not in df.columns
        ):

            return 0

        products = set()

        for value in (
            df[
                "Recommended Products"
            ]
            .dropna()
            .tolist()
        ):

            parsed = (
                self._split_recommendation_list(
                    value
                )
            )

            products.update(
                parsed
            )

        return len(
            products
        )

    # =========================================================
    # AVERAGE NUMBER OF RECOMMENDATIONS
    # =========================================================

    def _average_recommendation_count(
        self,
        df: pd.DataFrame
    ) -> float:

        if (
            "Recommended Products"
            not in df.columns
        ):

            return 0.0

        counts = []

        for value in (
            df[
                "Recommended Products"
            ]
            .dropna()
            .tolist()
        ):

            products = (
                self._split_recommendation_list(
                    value
                )
            )

            counts.append(
                len(
                    products
                )
            )

        if not counts:

            return 0.0

        return float(
            sum(
                counts
            )
            /
            len(
                counts
            )
        )

    # =========================================================
    # SPLIT PRODUCT LIST
    # =========================================================

    def _split_recommendation_list(
        self,
        value
    ) -> list:

        if pd.isna(
            value
        ):

            return []

        products = [
            item.strip()
            for item
            in str(
                value
            ).split(
                "|"
            )
            if item.strip()
        ]

        return products

    # =========================================================
    # TOP PRODUCT DISTRIBUTION
    # =========================================================

    def top_product_distribution(
        self,
        df: pd.DataFrame,
        top_n: int = 20
    ) -> pd.DataFrame:

        data = (
            self.prepare_data(
                df
            )
        )

        if (
            "Top Recommended Product"
            not in data.columns
        ):

            return pd.DataFrame(
                columns=[
                    "Product",
                    "Recommendations"
                ]
            )

        result = (
            data[
                "Top Recommended Product"
            ]
            .dropna()
            .value_counts()
            .head(
                top_n
            )
            .rename_axis(
                "Product"
            )
            .reset_index(
                name="Recommendations"
            )
        )

        return result

    # =========================================================
    # ALL RECOMMENDED PRODUCT FREQUENCY
    # =========================================================

    def all_product_distribution(
        self,
        df: pd.DataFrame,
        top_n: int = 20
    ) -> pd.DataFrame:

        data = (
            self.prepare_data(
                df
            )
        )

        if (
            "Recommended Products"
            not in data.columns
        ):

            return pd.DataFrame(
                columns=[
                    "Product",
                    "Recommendation Count"
                ]
            )

        products = []

        for value in (
            data[
                "Recommended Products"
            ]
            .dropna()
            .tolist()
        ):

            products.extend(
                self._split_recommendation_list(
                    value
                )
            )

        if not products:

            return pd.DataFrame(
                columns=[
                    "Product",
                    "Recommendation Count"
                ]
            )

        result = (
            pd.Series(
                products
            )
            .value_counts()
            .head(
                top_n
            )
            .rename_axis(
                "Product"
            )
            .reset_index(
                name="Recommendation Count"
            )
        )

        return result

    # =========================================================
    # RECOMMENDATION SOURCE DISTRIBUTION
    # =========================================================

    def source_distribution(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = (
            self.prepare_data(
                df
            )
        )

        if (
            "Recommendation Source"
            not in data.columns
        ):

            return pd.DataFrame(
                columns=[
                    "Recommendation Source",
                    "Customers"
                ]
            )

        result = (
            data[
                "Recommendation Source"
            ]
            .dropna()
            .value_counts()
            .rename_axis(
                "Recommendation Source"
            )
            .reset_index(
                name="Customers"
            )
        )

        return result

    # =========================================================
    # TOP PRODUCT BY CUSTOMER SEGMENT
    # =========================================================

    def products_by_segment(
        self,
        df: pd.DataFrame,
        top_n_per_segment: int = 5
    ) -> pd.DataFrame:

        data = (
            self.prepare_data(
                df
            )
        )

        required = [
            "Customer Segment",
            "Top Recommended Product"
        ]

        if any(
            column not in data.columns
            for column in required
        ):

            return pd.DataFrame()

        valid = (
            data
            .dropna(
                subset=[
                    "Customer Segment",
                    "Top Recommended Product"
                ]
            )
            .copy()
        )

        if valid.empty:

            return pd.DataFrame()

        result = (
            valid
            .groupby(
                [
                    "Customer Segment",
                    "Top Recommended Product"
                ]
            )
            .size()
            .reset_index(
                name="Customers"
            )
        )

        result[
            "Segment Product Rank"
        ] = (
            result
            .groupby(
                "Customer Segment"
            )[
                "Customers"
            ]
            .rank(
                method="first",
                ascending=False
            )
        )

        result = (
            result[
                result[
                    "Segment Product Rank"
                ]
                <=
                top_n_per_segment
            ]
            .sort_values(
                by=[
                    "Customer Segment",
                    "Segment Product Rank"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        result[
            "Segment Product Rank"
        ] = (
            result[
                "Segment Product Rank"
            ]
            .astype(int)
        )

        return result

    # =========================================================
    # TOP PRODUCTS BY CLV BAND
    # =========================================================

    def products_by_clv_band(
        self,
        df: pd.DataFrame,
        top_n_per_band: int = 5
    ) -> pd.DataFrame:

        data = (
            self.prepare_data(
                df
            )
        )

        required = [
            "CLV Value Band",
            "Top Recommended Product"
        ]

        if any(
            column not in data.columns
            for column in required
        ):

            return pd.DataFrame()

        valid = (
            data
            .dropna(
                subset=[
                    "CLV Value Band",
                    "Top Recommended Product"
                ]
            )
            .copy()
        )

        if valid.empty:

            return pd.DataFrame()

        result = (
            valid
            .groupby(
                [
                    "CLV Value Band",
                    "Top Recommended Product"
                ]
            )
            .size()
            .reset_index(
                name="Customers"
            )
        )

        result[
            "Band Product Rank"
        ] = (
            result
            .groupby(
                "CLV Value Band"
            )[
                "Customers"
            ]
            .rank(
                method="first",
                ascending=False
            )
        )

        result = (
            result[
                result[
                    "Band Product Rank"
                ]
                <=
                top_n_per_band
            ]
            .sort_values(
                by=[
                    "CLV Value Band",
                    "Band Product Rank"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        result[
            "Band Product Rank"
        ] = (
            result[
                "Band Product Rank"
            ]
            .astype(int)
        )

        return result

    # =========================================================
    # TOP PRODUCTS BY CHURN RISK
    # =========================================================

    def products_by_churn_risk(
        self,
        df: pd.DataFrame,
        top_n_per_risk: int = 5
    ) -> pd.DataFrame:

        data = (
            self.prepare_data(
                df
            )
        )

        required = [
            "Churn Risk",
            "Top Recommended Product"
        ]

        if any(
            column not in data.columns
            for column in required
        ):

            return pd.DataFrame()

        valid = (
            data
            .dropna(
                subset=[
                    "Churn Risk",
                    "Top Recommended Product"
                ]
            )
            .copy()
        )

        if valid.empty:

            return pd.DataFrame()

        result = (
            valid
            .groupby(
                [
                    "Churn Risk",
                    "Top Recommended Product"
                ]
            )
            .size()
            .reset_index(
                name="Customers"
            )
        )

        result[
            "Risk Product Rank"
        ] = (
            result
            .groupby(
                "Churn Risk"
            )[
                "Customers"
            ]
            .rank(
                method="first",
                ascending=False
            )
        )

        result = (
            result[
                result[
                    "Risk Product Rank"
                ]
                <=
                top_n_per_risk
            ]
            .sort_values(
                by=[
                    "Churn Risk",
                    "Risk Product Rank"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        result[
            "Risk Product Rank"
        ] = (
            result[
                "Risk Product Rank"
            ]
            .astype(int)
        )

        return result

    # =========================================================
    # CUSTOMER RECOMMENDATION DETAILS
    # =========================================================

    def customer_recommendations(
        self,
        df: pd.DataFrame,
        customer_id
    ) -> pd.DataFrame:

        data = (
            self.prepare_data(
                df
            )
        )

        customer = (
            data[
                data[
                    CUSTOMER_ID
                ]
                ==
                customer_id
            ]
        )

        if customer.empty:

            return pd.DataFrame(
                columns=[
                    "Rank",
                    "Product"
                ]
            )

        row = (
            customer
            .iloc[0]
        )

        products = (
            self._split_recommendation_list(
                row.get(
                    "Recommended Products"
                )
            )
        )

        codes = (
            self._split_recommendation_list(
                row.get(
                    "Recommended Stock Codes"
                )
            )
        )

        max_length = max(
            len(
                products
            ),
            len(
                codes
            )
        )

        if max_length == 0:

            return pd.DataFrame(
                columns=[
                    "Rank",
                    "Stock Code",
                    "Product"
                ]
            )

        rows = []

        for index in range(
            max_length
        ):

            product = (
                products[
                    index
                ]
                if index
                <
                len(
                    products
                )
                else ""
            )

            stock_code = (
                codes[
                    index
                ]
                if index
                <
                len(
                    codes
                )
                else ""
            )

            rows.append(
                {
                    "Rank":
                        index + 1,

                    "Stock Code":
                        stock_code,

                    "Product":
                        product
                }
            )

        return pd.DataFrame(
            rows
        )

    # =========================================================
    # CUSTOMER SUMMARY
    # =========================================================

    def customer_summary(
        self,
        df: pd.DataFrame,
        customer_id
    ):

        data = (
            self.prepare_data(
                df
            )
        )

        customer = (
            data[
                data[
                    CUSTOMER_ID
                ]
                ==
                customer_id
            ]
        )

        if customer.empty:

            return None

        return (
            customer
            .iloc[0]
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
            "Recommendation Intelligence"
        )

        st.caption(
            "Analyze personalized product recommendation "
            "coverage, product patterns and customer-level "
            "Top-N recommendations."
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
                "Recommendation Coverage",
                (
                    f"{metrics['recommendation_coverage']:.2f}%"
                )
            )

        with col2:

            st.metric(
                "Customers With Recommendations",
                f"{metrics['customers_with_recommendations']:,}"
            )

        with col3:

            st.metric(
                "Unique Recommended Products",
                f"{metrics['unique_recommended_products']:,}"
            )

        with col4:

            st.metric(
                "Avg Recommendations / Customer",
                (
                    f"{metrics['average_recommendations_per_customer']:.2f}"
                )
            )

        col5, col6, col7 = (
            st.columns(
                3
            )
        )

        with col5:

            st.metric(
                "Total Customers",
                f"{metrics['total_customers']:,}"
            )

        with col6:

            st.metric(
                "Customers Without Recommendations",
                f"{metrics['customers_without_recommendations']:,}"
            )

        with col7:

            st.metric(
                "Unique Top Products",
                f"{metrics['unique_top_products']:,}"
            )

        st.divider()

        # =====================================================
        # TOP PRODUCTS
        # =====================================================

        left, right = (
            st.columns(
                2
            )
        )

        with left:

            st.subheader(
                "Most Common Top Recommendations"
            )

            top_products = (
                self.top_product_distribution(
                    data,
                    top_n=15
                )
            )

            if top_products.empty:

                st.info(
                    "Top recommendation data "
                    "is not available."
                )

            else:

                top_chart = (
                    px.bar(
                        top_products,
                        x=
                            "Recommendations",
                        y=
                            "Product",
                        orientation=
                            "h",
                        text=
                            "Recommendations",
                        title=
                            "Most Frequent Rank-1 Products"
                    )
                )

                top_chart.update_layout(
                    yaxis={
                        "categoryorder":
                            "total ascending"
                    }
                )

                st.plotly_chart(
                    top_chart,
                    use_container_width=True
                )

        with right:

            st.subheader(
                "Most Recommended Products Across Top-N"
            )

            all_products = (
                self.all_product_distribution(
                    data,
                    top_n=15
                )
            )

            if all_products.empty:

                st.info(
                    "Complete Top-N recommendation "
                    "lists are not available."
                )

            else:

                all_chart = (
                    px.bar(
                        all_products,
                        x=
                            "Recommendation Count",
                        y=
                            "Product",
                        orientation=
                            "h",
                        text=
                            "Recommendation Count",
                        title=
                            "Most Frequent Products Across Top-N"
                    )
                )

                all_chart.update_layout(
                    yaxis={
                        "categoryorder":
                            "total ascending"
                    }
                )

                st.plotly_chart(
                    all_chart,
                    use_container_width=True
                )

        st.divider()

        # =====================================================
        # SCORE + SOURCE
        # =====================================================

        col1, col2 = (
            st.columns(
                2
            )
        )

        with col1:

            st.subheader(
                "Top Recommendation Score Distribution"
            )

            if (
                "Top Recommendation Score"
                in data.columns
            ):

                score_data = (
                    data[
                        data[
                            "Top Recommendation Score"
                        ]
                        .notna()
                    ]
                )

                if not score_data.empty:

                    score_chart = (
                        px.histogram(
                            score_data,
                            x=
                                "Top Recommendation Score",
                            nbins=
                                30,
                            title=
                                "Recommendation Ranking Scores"
                        )
                    )

                    st.plotly_chart(
                        score_chart,
                        use_container_width=True
                    )

                else:

                    st.info(
                        "Recommendation score data "
                        "is not available."
                    )

            else:

                st.info(
                    "Recommendation score data "
                    "is not available."
                )

        with col2:

            st.subheader(
                "Recommendation Source"
            )

            sources = (
                self.source_distribution(
                    data
                )
            )

            if sources.empty:

                st.info(
                    "Recommendation source information "
                    "is not available."
                )

            else:

                source_chart = (
                    px.pie(
                        sources,
                        names=
                            "Recommendation Source",
                        values=
                            "Customers",
                        title=
                            "Recommendation Strategy Usage"
                    )
                )

                st.plotly_chart(
                    source_chart,
                    use_container_width=True
                )

        st.caption(
            "Recommendation scores are model-specific "
            "ranking signals. They are not calibrated "
            "purchase probabilities."
        )

        st.divider()

        # =====================================================
        # PRODUCTS BY SEGMENT
        # =====================================================

        st.subheader(
            "Top Recommended Products by Customer Segment"
        )

        segment_products = (
            self.products_by_segment(
                data,
                top_n_per_segment=5
            )
        )

        if segment_products.empty:

            st.info(
                "Customer Segment recommendation "
                "analysis is not available."
            )

        else:

            segment_chart = (
                px.bar(
                    segment_products,
                    x=
                        "Customer Segment",
                    y=
                        "Customers",
                    color=
                        "Top Recommended Product",
                    barmode=
                        "group",
                    hover_data=[
                        "Segment Product Rank"
                    ],
                    title=
                        "Top Products Within Each Segment"
                )
            )

            st.plotly_chart(
                segment_chart,
                use_container_width=True
            )

            st.dataframe(
                segment_products,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        # =====================================================
        # CLV + CHURN ANALYSIS
        # =====================================================

        left, right = (
            st.columns(
                2
            )
        )

        with left:

            st.subheader(
                "Recommendations by CLV Value Band"
            )

            band_products = (
                self.products_by_clv_band(
                    data,
                    top_n_per_band=5
                )
            )

            if band_products.empty:

                st.info(
                    "CLV Value Band recommendation "
                    "analysis is not available."
                )

            else:

                band_chart = (
                    px.bar(
                        band_products,
                        x=
                            "CLV Value Band",
                        y=
                            "Customers",
                        color=
                            "Top Recommended Product",
                        barmode=
                            "group",
                        title=
                            "Top Products by Customer Value Band"
                    )
                )

                st.plotly_chart(
                    band_chart,
                    use_container_width=True
                )

        with right:

            st.subheader(
                "Recommendations by Churn Risk"
            )

            risk_products = (
                self.products_by_churn_risk(
                    data,
                    top_n_per_risk=5
                )
            )

            if risk_products.empty:

                st.info(
                    "Churn Risk recommendation "
                    "analysis is not available."
                )

            else:

                risk_chart = (
                    px.bar(
                        risk_products,
                        x=
                            "Churn Risk",
                        y=
                            "Customers",
                        color=
                            "Top Recommended Product",
                        barmode=
                            "group",
                        title=
                            "Top Products by Churn Risk"
                    )
                )

                st.plotly_chart(
                    risk_chart,
                    use_container_width=True
                )

        st.divider()

        # =====================================================
        # CUSTOMER RECOMMENDATION EXPLORER
        # =====================================================

        st.subheader(
            "Customer Recommendation Explorer"
        )

        customer_options = (
            data[
                CUSTOMER_ID
            ]
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        selected_customer = (
            st.selectbox(
                "Select Customer ID",
                options=
                    customer_options,
                key=
                    "recommendation_customer_explorer"
            )
        )

        customer = (
            self.customer_summary(
                data,
                selected_customer
            )
        )

        if customer is None:

            st.warning(
                "Customer not found."
            )

        else:

            col1, col2, col3, col4 = (
                st.columns(
                    4
                )
            )

            with col1:

                st.metric(
                    "Customer ID",
                    customer.get(
                        CUSTOMER_ID,
                        ""
                    )
                )

            with col2:

                st.metric(
                    "Segment",
                    customer.get(
                        "Customer Segment",
                        "N/A"
                    )
                )

            with col3:

                st.metric(
                    "Churn Risk",
                    customer.get(
                        "Churn Risk",
                        "N/A"
                    )
                )

            with col4:

                st.metric(
                    "CLV Value Band",
                    customer.get(
                        "CLV Value Band",
                        "N/A"
                    )
                )

            # -------------------------------------------------
            # Top recommendation
            # -------------------------------------------------

            st.write(
                "**Top Recommended Product:**",
                customer.get(
                    "Top Recommended Product",
                    "Not Available"
                )
            )

            if (
                "Recommendation Source"
                in data.columns
            ):

                st.write(
                    "**Recommendation Source:**",
                    customer.get(
                        "Recommendation Source",
                        "Not Available"
                    )
                )

            if (
                "Top Recommendation Score"
                in data.columns
                and
                pd.notna(
                    customer.get(
                        "Top Recommendation Score"
                    )
                )
            ):

                st.write(
                    "**Top Recommendation Score:** "
                    f"{float(customer['Top Recommendation Score']):.4f}"
                )

            recommendations = (
                self.customer_recommendations(
                    data,
                    selected_customer
                )
            )

            if recommendations.empty:

                st.info(
                    "No Top-N recommendations available "
                    "for this customer."
                )

            else:

                st.dataframe(
                    recommendations,
                    use_container_width=True,
                    hide_index=True
                )

        st.divider()

        # =====================================================
        # COMPLETE RECOMMENDATION TABLE
        # =====================================================

        st.subheader(
            "Customer Recommendation Summary"
        )

        desired_columns = [
            CUSTOMER_ID,
            "Customer Segment",
            "Churn Risk",
            "CLV Value Band",
            "Top Recommended Stock Code",
            "Top Recommended Product",
            "Top Recommendation Score",
            "Recommendation Source",
            "Recommended Products"
        ]

        available_columns = [
            column
            for column in desired_columns
            if column in data.columns
        ]

        recommendation_table = (
            data[
                available_columns
            ]
            .copy()
        )

        st.dataframe(
            recommendation_table,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "The current dashboard is based on the "
            "persisted recommendation output generated "
            "for the current production mode. "
            "A direct next-purchase versus discovery "
            "comparison would require storing predictions "
            "from both modes separately."
        )