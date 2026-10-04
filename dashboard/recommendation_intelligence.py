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

        return data


    # =========================================================
    # SPLIT RECOMMENDATION LIST
    # =========================================================

    def _split_recommendation_list(
        self,
        value
    ) -> list:

        if pd.isna(
            value
        ):

            return []

        return [
            item.strip()
            for item
            in str(
                value
            ).split(
                "|"
            )
            if item.strip()
        ]


    # =========================================================
    # UNIQUE RECOMMENDED PRODUCTS
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

            products.update(
                self._split_recommendation_list(
                    value
                )
            )

        return len(
            products
        )


    # =========================================================
    # AVERAGE RECOMMENDATIONS PER CUSTOMER
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

            counts.append(
                len(
                    self._split_recommendation_list(
                        value
                    )
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
            ]
            .nunique()
        )

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

        customers_without_recommendations = (
            total_customers
            -
            customers_with_recommendations
        )

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

        if (
            "Top Recommendation Score"
            in data.columns
        ):

            score_series = (
                data[
                    "Top Recommendation Score"
                ]
                .dropna()
            )

            if score_series.empty:

                average_score = 0.0

            else:

                average_score = float(
                    score_series.mean()
                )

        else:

            average_score = 0.0

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
                self._count_unique_recommended_products(
                    data
                ),

            "average_top_recommendation_score":
                average_score,

            "average_recommendations_per_customer":
                self._average_recommendation_count(
                    data
                )
        }


    # =========================================================
    # TOP PRODUCT DISTRIBUTION
    # =========================================================

    def top_product_distribution(
        self,
        df: pd.DataFrame,
        top_n: int = 20
    ) -> pd.DataFrame:

        data = self.prepare_data(
            df
        )

        if (
            "Top Recommended Product"
            not in data.columns
        ):

            return pd.DataFrame()

        return (
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


    # =========================================================
    # ALL PRODUCT DISTRIBUTION
    # =========================================================

    def all_product_distribution(
        self,
        df: pd.DataFrame,
        top_n: int = 20
    ) -> pd.DataFrame:

        data = self.prepare_data(
            df
        )

        if (
            "Recommended Products"
            not in data.columns
        ):

            return pd.DataFrame()

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

            return pd.DataFrame()

        return (
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


    # =========================================================
    # SOURCE DISTRIBUTION
    # =========================================================

    def source_distribution(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = self.prepare_data(
            df
        )

        if (
            "Recommendation Source"
            not in data.columns
        ):

            return pd.DataFrame()

        return (
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


    # =========================================================
    # PRODUCTS BY SEGMENT
    # =========================================================

    def products_by_segment(
        self,
        df: pd.DataFrame,
        top_n_per_segment: int = 5
    ) -> pd.DataFrame:

        data = self.prepare_data(
            df
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
                subset=required
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
    # PRODUCTS BY CLV BAND
    # =========================================================

    def products_by_clv_band(
        self,
        df: pd.DataFrame,
        top_n_per_band: int = 5
    ) -> pd.DataFrame:

        data = self.prepare_data(
            df
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
                subset=required
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
    # PRODUCTS BY CHURN RISK
    # =========================================================

    def products_by_churn_risk(
        self,
        df: pd.DataFrame,
        top_n_per_risk: int = 5
    ) -> pd.DataFrame:

        data = self.prepare_data(
            df
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
                subset=required
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
    # CUSTOMER RECOMMENDATIONS
    # =========================================================

    def customer_recommendations(
        self,
        df: pd.DataFrame,
        customer_id
    ) -> pd.DataFrame:

        data = self.prepare_data(
            df
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

            return pd.DataFrame()

        row = customer.iloc[0]

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
            len(products),
            len(codes)
        )

        if max_length == 0:

            return pd.DataFrame()

        rows = []

        for index in range(
            max_length
        ):

            rows.append(
                {
                    "Rank":
                        index + 1,

                    "Stock Code":
                        (
                            codes[index]
                            if index < len(codes)
                            else ""
                        ),

                    "Product":
                        (
                            products[index]
                            if index < len(products)
                            else ""
                        )
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

        data = self.prepare_data(
            df
        )

        result = (
            data[
                data[
                    CUSTOMER_ID
                ]
                ==
                customer_id
            ]
        )

        if result.empty:

            return None

        return result.iloc[0]


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
                    "Recommendation Coverage",

                value=
                    (
                        f"{metrics['recommendation_coverage']:.1f}%"
                    ),

                icon=
                    "🎯",

                card_class=
                    "card-blue"
            )

        with row1[1]:

            render_color_card(
                title=
                    "Customers Recommended",

                value=
                    (
                        f"{metrics['customers_with_recommendations']:,}"
                    ),

                icon=
                    "👥",

                card_class=
                    "card-green"
            )

        with row1[2]:

            render_color_card(
                title=
                    "Unique Products",

                value=
                    (
                        f"{metrics['unique_recommended_products']:,}"
                    ),

                icon=
                    "🛍️",

                card_class=
                    "card-purple"
            )

        with row1[3]:

            render_color_card(
                title=
                    "Avg Products / Customer",

                value=
                    (
                        f"{metrics['average_recommendations_per_customer']:.1f}"
                    ),

                icon=
                    "📦",

                card_class=
                    "card-blue"
            )

        st.write("")

        row2 = st.columns(
            3
        )

        with row2[0]:

            render_color_card(
                title=
                    "Without Recommendations",

                value=
                    (
                        f"{metrics['customers_without_recommendations']:,}"
                    ),

                icon=
                    "⚠️",

                card_class=
                    "card-red"
            )

        with row2[1]:

            render_color_card(
                title=
                    "Unique Rank-1 Products",

                value=
                    (
                        f"{metrics['unique_top_products']:,}"
                    ),

                icon=
                    "🏆",

                card_class=
                    "card-purple"
            )

        with row2[2]:

            render_color_card(
                title=
                    "Avg Ranking Score",

                value=
                    (
                        f"{metrics['average_top_recommendation_score']:.3f}"
                    ),

                icon=
                    "📊",

                card_class=
                    "card-green"
            )


    # =========================================================
    # COVERAGE GAUGE
    # =========================================================

    def _coverage_gauge(
        self,
        metrics: dict
    ):

        value = float(
            metrics[
                "recommendation_coverage"
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
                        "Recommendation Coverage"
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
                            "#4f46e5"
                    },

                    "steps": [
                        {
                            "range": [
                                0,
                                50
                            ],
                            "color":
                                "#fee2e2"
                        },

                        {
                            "range": [
                                50,
                                80
                            ],
                            "color":
                                "#fef3c7"
                        },

                        {
                            "range": [
                                80,
                                100
                            ],
                            "color":
                                "#dcfce7"
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
    # SOURCE DONUT
    # =========================================================

    def _source_donut(
        self,
        df: pd.DataFrame
    ):

        sources = self.source_distribution(
            df
        )

        if sources.empty:

            return None

        figure = px.pie(
            sources,

            names=
                "Recommendation Source",

            values=
                "Customers",

            hole=
                0.58,

            title=
                "Recommendation Source Mix"
        )

        figure.update_traces(
            textinfo=
                "percent+label"
        )

        figure.update_layout(
            height=390,
            legend_title_text=""
        )

        return figure


    # =========================================================
    # PRODUCT TREEMAP
    # =========================================================

    def _product_treemap(
        self,
        df: pd.DataFrame,
        top_n: int = 25
    ):

        products = (
            self.all_product_distribution(
                df,
                top_n=top_n
            )
        )

        if products.empty:

            return None

        figure = px.treemap(
            products,

            path=[
                "Product"
            ],

            values=
                "Recommendation Count",

            color=
                "Recommendation Count",

            color_continuous_scale=
                "Viridis",

            title=
                "Top Recommended Product Landscape"
        )

        figure.update_layout(
            height=500
        )

        return figure


    # =========================================================
    # PRODUCT CARDS
    # =========================================================

    def _render_recommendation_cards(
        self,
        recommendations: pd.DataFrame
    ):

        if recommendations.empty:

            st.info(
                "No Top-N recommendations "
                "available for this customer."
            )

            return

        columns = st.columns(
            min(
                len(recommendations),
                5
            )
        )

        for index, (
            _,
            row
        ) in enumerate(
            recommendations
            .head(5)
            .iterrows()
        ):

            with columns[index]:

                rank = row.get(
                    "Rank",
                    index + 1
                )

                product = row.get(
                    "Product",
                    "Unknown Product"
                )

                stock_code = row.get(
                    "Stock Code",
                    ""
                )

                st.markdown(
                    (
                        '<div style="'
                        'background:linear-gradient('
                        '135deg,#4f46e5,#7c3aed);'
                        'padding:18px;'
                        'border-radius:16px;'
                        'color:white;'
                        'min-height:180px;'
                        'box-shadow:0 10px 25px '
                        'rgba(79,70,229,0.20);'
                        '">'
                        f'<div style="font-size:0.8rem;'
                        f'opacity:0.8;">RANK #{rank}</div>'
                        f'<div style="font-size:1rem;'
                        f'font-weight:700;'
                        f'margin-top:10px;">'
                        f'{product}'
                        f'</div>'
                        f'<div style="font-size:0.8rem;'
                        f'margin-top:14px;'
                        f'opacity:0.85;">'
                        f'Stock Code: {stock_code}'
                        f'</div>'
                        '</div>'
                    ),
                    unsafe_allow_html=True
                )


    # =========================================================
    # STYLED SUMMARY TABLE
    # =========================================================

    def _styled_summary_table(
        self,
        df: pd.DataFrame
    ):

        desired_columns = [
            CUSTOMER_ID,
            "Customer Segment",
            "Churn Risk",
            "CLV Value Band",
            "Top Recommended Product",
            "Top Recommendation Score",
            "Recommendation Source",
            "Recommended Products"
        ]

        available_columns = [
            column
            for column in desired_columns
            if column in df.columns
        ]

        table = (
            df[
                available_columns
            ]
            .copy()
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

        metrics = self.calculate_metrics(
            data
        )

        st.markdown(
            "## 🎯 Recommendation Intelligence"
        )

        st.caption(
            "Interactive analysis of personalized product "
            "recommendations, coverage, product patterns, "
            "customer value and recommendation strategy usage."
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

        tab1, tab2, tab3, tab4 = (
            st.tabs(
                [
                    "📊 Recommendation Overview",
                    "🛍️ Product Intelligence",
                    "🧩 Customer Groups",
                    "👤 Customer Explorer"
                ]
            )
        )

        # =====================================================
        # TAB 1 — OVERVIEW
        # =====================================================

        with tab1:

            col1, col2 = st.columns(
                2
            )

            with col1:

                gauge = self._coverage_gauge(
                    metrics
                )

                st.plotly_chart(
                    gauge,
                    use_container_width=True
                )

            with col2:

                source_donut = (
                    self._source_donut(
                        data
                    )
                )

                if source_donut is None:

                    st.info(
                        "Recommendation source data "
                        "is not available."
                    )

                else:

                    st.plotly_chart(
                        source_donut,
                        use_container_width=True
                    )

            st.markdown(
                "### 🏆 Most Common Rank-1 Recommendations"
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

                chart = px.bar(
                    top_products,

                    x=
                        "Recommendations",

                    y=
                        "Product",

                    orientation=
                        "h",

                    color=
                        "Recommendations",

                    text=
                        "Recommendations",

                    color_continuous_scale=
                        "Viridis",

                    title=
                        "Most Frequent Rank-1 Products"
                )

                chart.update_layout(
                    yaxis={
                        "categoryorder":
                            "total ascending"
                    },

                    coloraxis_showscale=
                        False,

                    height=
                        550
                )

                st.plotly_chart(
                    chart,
                    use_container_width=True
                )

            if (
                "Top Recommendation Score"
                in data.columns
            ):

                st.markdown(
                    "### 📈 Recommendation Score Distribution"
                )

                score_data = (
                    data[
                        data[
                            "Top Recommendation Score"
                        ]
                        .notna()
                    ]
                )

                if not score_data.empty:

                    score_chart = px.histogram(
                        score_data,

                        x=
                            "Top Recommendation Score",

                        nbins=
                            35,

                        color=(
                            "Recommendation Source"
                            if
                            "Recommendation Source"
                            in score_data.columns
                            else
                            None
                        ),

                        title=
                            "Top Recommendation Ranking Scores"
                    )

                    st.plotly_chart(
                        score_chart,
                        use_container_width=True
                    )

            st.info(
                "Recommendation scores are ranking signals. "
                "They are not calibrated probabilities "
                "of purchase."
            )


        # =====================================================
        # TAB 2 — PRODUCTS
        # =====================================================

        with tab2:

            st.markdown(
                "### 🛍️ Recommended Product Landscape"
            )

            treemap = self._product_treemap(
                data,
                top_n=30
            )

            if treemap is not None:

                st.plotly_chart(
                    treemap,
                    use_container_width=True
                )

            all_products = (
                self.all_product_distribution(
                    data,
                    top_n=20
                )
            )

            if not all_products.empty:

                chart = px.bar(
                    all_products,

                    x=
                        "Recommendation Count",

                    y=
                        "Product",

                    orientation=
                        "h",

                    color=
                        "Recommendation Count",

                    text=
                        "Recommendation Count",

                    color_continuous_scale=
                        "Plasma",

                    title=
                        "Most Recommended Products Across Top-N"
                )

                chart.update_layout(
                    yaxis={
                        "categoryorder":
                            "total ascending"
                    },

                    coloraxis_showscale=
                        False,

                    height=
                        600
                )

                st.plotly_chart(
                    chart,
                    use_container_width=True
                )


        # =====================================================
        # TAB 3 — CUSTOMER GROUPS
        # =====================================================

        with tab3:

            st.markdown(
                "### 🧩 Recommendations by Customer Segment"
            )

            segment_products = (
                self.products_by_segment(
                    data,
                    top_n_per_segment=5
                )
            )

            if segment_products.empty:

                st.info(
                    "Segment recommendation analysis "
                    "is not available."
                )

            else:

                segment_chart = px.bar(
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
                        "Top Products by Customer Segment"
                )

                st.plotly_chart(
                    segment_chart,
                    use_container_width=True
                )

            col1, col2 = st.columns(
                2
            )

            with col1:

                st.markdown(
                    "#### 💎 Products by CLV Band"
                )

                band_products = (
                    self.products_by_clv_band(
                        data,
                        top_n_per_band=5
                    )
                )

                if band_products.empty:

                    st.info(
                        "CLV-band recommendation "
                        "analysis is not available."
                    )

                else:

                    band_chart = px.bar(
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
                            "Top Products by Value Band"
                    )

                    st.plotly_chart(
                        band_chart,
                        use_container_width=True
                    )

            with col2:

                st.markdown(
                    "#### ⚠️ Products by Churn Risk"
                )

                risk_products = (
                    self.products_by_churn_risk(
                        data,
                        top_n_per_risk=5
                    )
                )

                if risk_products.empty:

                    st.info(
                        "Churn-risk recommendation "
                        "analysis is not available."
                    )

                else:

                    risk_chart = px.bar(
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

                    st.plotly_chart(
                        risk_chart,
                        use_container_width=True
                    )


        # =====================================================
        # TAB 4 — CUSTOMER EXPLORER
        # =====================================================

        with tab4:

            st.markdown(
                "### 👤 Customer Recommendation Explorer"
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
                        "premium_recommendation_customer"
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

                row = st.columns(
                    4
                )

                with row[0]:

                    render_color_card(
                        title=
                            "Customer ID",

                        value=
                            customer.get(
                                CUSTOMER_ID,
                                ""
                            ),

                        icon=
                            "👤",

                        card_class=
                            "card-blue"
                    )

                with row[1]:

                    render_color_card(
                        title=
                            "Segment",

                        value=
                            customer.get(
                                "Customer Segment",
                                "N/A"
                            ),

                        icon=
                            "🧩",

                        card_class=
                            "card-purple"
                    )

                with row[2]:

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
                            "⚠️",

                        card_class=
                            "card-red"
                    )

                with row[3]:

                    render_color_card(
                        title=
                            "CLV Band",

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
                            "card-green"
                    )

                st.write("")

                st.markdown(
                    "### 🎯 Personalized Top Recommendations"
                )

                recommendations = (
                    self.customer_recommendations(
                        data,
                        selected_customer
                    )
                )

                self._render_recommendation_cards(
                    recommendations
                )

                st.write("")

                if (
                    "Recommendation Source"
                    in data.columns
                ):

                    st.info(
                        (
                            "Recommendation Source: "
                            f"**{customer.get('Recommendation Source', 'N/A')}**"
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

                    st.success(
                        (
                            "Top Ranking Score: "
                            f"**{float(customer['Top Recommendation Score']):.4f}**"
                        )
                    )

                st.markdown(
                    "### 📋 Complete Recommendation Summary"
                )

                self._styled_summary_table(
                    data[
                        data[
                            CUSTOMER_ID
                        ]
                        ==
                        selected_customer
                    ]
                )

            st.caption(
                "The current dashboard reflects the "
                "persisted production recommendation mode. "
                "A next-purchase versus discovery comparison "
                "requires separately generated outputs for "
                "both modes."
            )