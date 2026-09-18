import sys

import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger


class SegmentInterpreter:

    def create_segment_summary(
        self,
        clustered_df: pd.DataFrame
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Starting customer segment interpretation."
            )

            required_columns = [
                "CustomerID",
                "Recency",
                "Frequency",
                "Monetary",
                "TotalItems",
                "AverageOrderValue",
                "Tenure",
                "Cluster"
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in clustered_df.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"Missing required columns: "
                    f"{missing_columns}"
                )

            if clustered_df.empty:

                raise ValueError(
                    "Clustered customer dataset is empty."
                )

            data = clustered_df.copy()

            features = [
                "Recency",
                "Frequency",
                "Monetary",
                "TotalItems",
                "AverageOrderValue",
                "Tenure"
            ]

            # ---------------------------------
            # Overall customer averages
            # ---------------------------------

            overall_average = (
                data[features]
                .mean()
            )

            # ---------------------------------
            # Cluster averages
            # ---------------------------------

            cluster_average = (
                data
                .groupby(
                    "Cluster"
                )[features]
                .mean()
                .reset_index()
            )

            # ---------------------------------
            # Customer count
            # ---------------------------------

            cluster_count = (
                data
                .groupby(
                    "Cluster"
                )["CustomerID"]
                .nunique()
                .rename(
                    "CustomerCount"
                )
                .reset_index()
            )

            total_customers = (
                data["CustomerID"]
                .nunique()
            )

            cluster_count[
                "CustomerPercentage"
            ] = (
                cluster_count[
                    "CustomerCount"
                ]
                /
                total_customers
                *
                100
            )

            # ---------------------------------
            # Merge cluster information
            # ---------------------------------

            summary = (
                cluster_average
                .merge(
                    cluster_count,
                    on="Cluster",
                    how="left"
                )
            )

            # ---------------------------------
            # Behavioral flags
            # ---------------------------------

            summary[
                "RecentPurchase"
            ] = (
                summary["Recency"]
                <
                overall_average["Recency"]
            )

            summary[
                "HighFrequency"
            ] = (
                summary["Frequency"]
                >
                overall_average["Frequency"]
            )

            summary[
                "HighMonetary"
            ] = (
                summary["Monetary"]
                >
                overall_average["Monetary"]
            )

            summary[
                "HighVolume"
            ] = (
                summary["TotalItems"]
                >
                overall_average["TotalItems"]
            )

            summary[
                "HighAOV"
            ] = (
                summary["AverageOrderValue"]
                >
                overall_average[
                    "AverageOrderValue"
                ]
            )

            summary[
                "LongTenure"
            ] = (
                summary["Tenure"]
                >
                overall_average["Tenure"]
            )

            # ---------------------------------
            # Assign segment names
            # ---------------------------------

            summary[
                "SegmentName"
            ] = summary.apply(
                self._assign_segment_name,
                axis=1
            )

            # ---------------------------------
            # Business recommendation
            # ---------------------------------

            summary[
                "Recommendation"
            ] = summary[
                "SegmentName"
            ].apply(
                self._business_recommendation
            )

            # ---------------------------------
            # Round numeric values
            # ---------------------------------

            numeric_columns = [
                "Recency",
                "Frequency",
                "Monetary",
                "TotalItems",
                "AverageOrderValue",
                "Tenure",
                "CustomerPercentage"
            ]

            summary[
                numeric_columns
            ] = (
                summary[
                    numeric_columns
                ]
                .round(2)
            )

            logger.info(
                "Customer segment interpretation completed."
            )

            logger.info(
                f"Segment summary:\n"
                f"{summary}"
            )

            return summary

        except Exception as e:

            logger.error(
                "Customer segment interpretation failed."
            )

            raise CustomException(
                e,
                sys
            )

    def _assign_segment_name(
        self,
        row: pd.Series
    ) -> str:

        recent = (
            row["RecentPurchase"]
        )

        high_frequency = (
            row["HighFrequency"]
        )

        high_monetary = (
            row["HighMonetary"]
        )

        high_volume = (
            row["HighVolume"]
        )

        high_aov = (
            row["HighAOV"]
        )

        long_tenure = (
            row["LongTenure"]
        )

        # ---------------------------------
        # High Value Loyal
        # ---------------------------------

        if (
            recent
            and high_frequency
            and high_monetary
            and long_tenure
        ):

            return "High Value Loyal"

        # ---------------------------------
        # High Value Occasional
        # ---------------------------------

        if (
            high_monetary
            and high_aov
            and not high_frequency
        ):

            return "High Value Occasional"

        # ---------------------------------
        # Loyal Regular
        # ---------------------------------

        if (
            recent
            and high_frequency
            and long_tenure
        ):

            return "Loyal Regular"

        # ---------------------------------
        # New / Potential
        # ---------------------------------

        if (
            recent
            and not long_tenure
            and not high_frequency
        ):

            return "New / Potential"

        # ---------------------------------
        # At Risk
        # ---------------------------------

        if (
            not recent
            and (
                high_monetary
                or high_frequency
                or long_tenure
            )
        ):

            return "At Risk"

        # ---------------------------------
        # Low Engagement
        # ---------------------------------

        if (
            not recent
            and not high_frequency
            and not high_monetary
            and not high_volume
        ):

            return "Low Engagement"

        # ---------------------------------
        # Regular Customer
        # ---------------------------------

        return "Regular Customer"

    def _business_recommendation(
        self,
        segment_name: str
    ) -> str:

        recommendations = {

            "High Value Loyal":
                (
                    "Focus on retention, loyalty rewards, "
                    "premium benefits, early access, and "
                    "personalized cross-sell recommendations."
                ),

            "High Value Occasional":
                (
                    "Encourage more frequent purchases through "
                    "personalized reminders, targeted offers, "
                    "and premium product recommendations."
                ),

            "Loyal Regular":
                (
                    "Maintain engagement with loyalty programs, "
                    "bundles, repeat-purchase offers, and "
                    "relevant cross-sell campaigns."
                ),

            "New / Potential":
                (
                    "Use onboarding campaigns, welcome offers, "
                    "product education, and second-purchase "
                    "incentives to build long-term engagement."
                ),

            "At Risk":
                (
                    "Prioritize retention campaigns, win-back "
                    "offers, personalized outreach, and churn "
                    "prevention strategies."
                ),

            "Low Engagement":
                (
                    "Use low-cost reactivation campaigns, "
                    "discount testing, and targeted messaging "
                    "before investing heavily in retention."
                ),

            "Regular Customer":
                (
                    "Maintain consistent engagement and use "
                    "personalized recommendations to increase "
                    "purchase frequency and customer value."
                )
        }

        return recommendations.get(
            segment_name,
            "Monitor customer behavior and apply personalized engagement."
        )