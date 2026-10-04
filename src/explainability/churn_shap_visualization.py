import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


class ChurnSHAPVisualization:

    CUSTOMER_ID = "Customer ID"

    # =========================================================
    # OUTPUT DIRECTORY
    # =========================================================

    @staticmethod
    def create_output_directory(
        output_directory: str
    ) -> None:

        os.makedirs(
            output_directory,
            exist_ok=True
        )

    # =========================================================
    # GLOBAL IMPORTANCE BAR CHART
    # =========================================================

    def save_global_importance_plot(
        self,
        importance_df: pd.DataFrame,
        output_path: str
    ) -> None:

        required_columns = [
            "Feature",
            "Mean Absolute SHAP"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in importance_df.columns
        ]

        if missing_columns:

            raise ValueError(
                "Missing required columns: "
                f"{missing_columns}"
            )

        plot_data = (
            importance_df[
                required_columns
            ]
            .copy()
            .sort_values(
                by="Mean Absolute SHAP",
                ascending=True
            )
        )

        figure, axis = plt.subplots(
            figsize=(
                10,
                6
            )
        )

        axis.barh(
            plot_data[
                "Feature"
            ],
            plot_data[
                "Mean Absolute SHAP"
            ]
        )

        axis.set_title(
            "Churn Model — Global SHAP Feature Importance"
        )

        axis.set_xlabel(
            "Mean Absolute SHAP Value"
        )

        axis.set_ylabel(
            "Feature"
        )

        axis.grid(
            axis="x",
            alpha=0.25
        )

        figure.tight_layout()

        figure.savefig(
            output_path,
            dpi=160,
            bbox_inches="tight"
        )

        plt.close(
            figure
        )

    # =========================================================
    # SHAP BEESWARM
    # =========================================================

    def save_beeswarm_plot(
        self,
        shap_values,
        output_path: str,
        max_display: int = 10
    ) -> None:

        if shap_values is None:

            raise ValueError(
                "SHAP values cannot be None."
            )

        plt.figure(
            figsize=(
                10,
                7
            )
        )

        shap.plots.beeswarm(
            shap_values,
            max_display=max_display,
            show=False
        )

        plt.title(
            "Churn Model — SHAP Beeswarm"
        )

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=160,
            bbox_inches="tight"
        )

        plt.close()

    # =========================================================
    # CUSTOMER DRIVER BAR CHART
    # =========================================================

    def save_customer_driver_plot(
        self,
        explanation_df: pd.DataFrame,
        output_path: str,
        top_n: int = 6
    ) -> None:

        required_columns = [
            "Feature",
            "SHAP Value",
            "Absolute SHAP"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in explanation_df.columns
        ]

        if missing_columns:

            raise ValueError(
                "Missing required columns: "
                f"{missing_columns}"
            )

        plot_data = (
            explanation_df
            .sort_values(
                by="Absolute SHAP",
                ascending=False
            )
            .head(
                top_n
            )
            .sort_values(
                by="SHAP Value"
            )
            .copy()
        )

        figure, axis = plt.subplots(
            figsize=(
                10,
                6
            )
        )

        axis.barh(
            plot_data[
                "Feature"
            ],
            plot_data[
                "SHAP Value"
            ]
        )

        axis.axvline(
            0,
            linewidth=1
        )

        axis.set_title(
            "Customer Churn Prediction Drivers"
        )

        axis.set_xlabel(
            "SHAP Contribution"
        )

        axis.set_ylabel(
            "Feature"
        )

        axis.grid(
            axis="x",
            alpha=0.25
        )

        figure.tight_layout()

        figure.savefig(
            output_path,
            dpi=160,
            bbox_inches="tight"
        )

        plt.close(
            figure
        )

    # =========================================================
    # CUSTOMER WATERFALL
    # =========================================================

    def save_customer_waterfall_plot(
        self,
        shap_explanation,
        output_path: str,
        max_display: int = 10
    ) -> None:

        if shap_explanation is None:

            raise ValueError(
                "SHAP explanation cannot be None."
            )

        # One customer is required for waterfall
        if (
            hasattr(
                shap_explanation,
                "values"
            )
            and
            np.asarray(
                shap_explanation.values
            ).ndim
            > 1
        ):

            customer_explanation = (
                shap_explanation[
                    0
                ]
            )

        else:

            customer_explanation = (
                shap_explanation
            )

        plt.figure(
            figsize=(
                10,
                7
            )
        )

        shap.plots.waterfall(
            customer_explanation,
            max_display=max_display,
            show=False
        )

        plt.title(
            "Customer Churn Prediction — SHAP Waterfall"
        )

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=160,
            bbox_inches="tight"
        )

        plt.close()

    # =========================================================
    # DASHBOARD-READY LONG-FORM SHAP DATA
    # =========================================================

    def build_dashboard_explanation_data(
        self,
        customer_ids,
        X: pd.DataFrame,
        shap_values
    ) -> pd.DataFrame:

        if len(
            customer_ids
        ) != len(
            X
        ):

            raise ValueError(
                "Customer IDs and feature rows "
                "must have equal length."
            )

        values = np.asarray(
            shap_values.values
        )

        # -----------------------------------------------------
        # Binary classification safety
        # -----------------------------------------------------

        if values.ndim == 3:

            if values.shape[-1] < 2:

                raise ValueError(
                    "Unsupported SHAP output shape: "
                    f"{values.shape}"
                )

            values = values[
                :,
                :,
                1
            ]

        if values.ndim != 2:

            raise ValueError(
                "Expected 2D SHAP values. "
                f"Received shape: {values.shape}"
            )

        if values.shape != X.shape:

            raise ValueError(
                "SHAP value dimensions do not "
                "match feature data."
            )

        records = []

        for row_position in range(
            len(X)
        ):

            customer_id = (
                customer_ids[
                    row_position
                ]
            )

            for feature_position, feature in enumerate(
                X.columns
            ):

                shap_value = float(
                    values[
                        row_position,
                        feature_position
                    ]
                )

                feature_value = float(
                    X.iloc[
                        row_position,
                        feature_position
                    ]
                )

                records.append(
                    {
                        self.CUSTOMER_ID:
                            customer_id,

                        "Feature":
                            feature,

                        "Feature Value":
                            feature_value,

                        "SHAP Value":
                            shap_value,

                        "Absolute SHAP":
                            abs(
                                shap_value
                            ),

                        "Impact Direction":
                            (
                                "Increases Churn Prediction"
                                if shap_value > 0
                                else
                                "Decreases Churn Prediction"
                                if shap_value < 0
                                else
                                "Neutral"
                            )
                    }
                )

        result = pd.DataFrame(
            records
        )

        result[
            "Impact Rank"
        ] = (
            result
            .groupby(
                self.CUSTOMER_ID
            )[
                "Absolute SHAP"
            ]
            .rank(
                method="first",
                ascending=False
            )
            .astype(int)
        )

        return result

    # =========================================================
    # TOP DRIVER SUMMARY
    # =========================================================

    def build_customer_driver_summary(
        self,
        explanation_df: pd.DataFrame,
        top_n: int = 3
    ) -> dict:

        required_columns = [
            "Feature",
            "Feature Value",
            "SHAP Value",
            "Absolute SHAP"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in explanation_df.columns
        ]

        if missing_columns:

            raise ValueError(
                "Missing required columns: "
                f"{missing_columns}"
            )

        increasing = (
            explanation_df[
                explanation_df[
                    "SHAP Value"
                ]
                >
                0
            ]
            .sort_values(
                by="Absolute SHAP",
                ascending=False
            )
            .head(
                top_n
            )
        )

        decreasing = (
            explanation_df[
                explanation_df[
                    "SHAP Value"
                ]
                <
                0
            ]
            .sort_values(
                by="Absolute SHAP",
                ascending=False
            )
            .head(
                top_n
            )
        )

        return {
            "top_churn_drivers":
                increasing[
                    [
                        "Feature",
                        "Feature Value",
                        "SHAP Value"
                    ]
                ]
                .to_dict(
                    orient="records"
                ),

            "top_retention_drivers":
                decreasing[
                    [
                        "Feature",
                        "Feature Value",
                        "SHAP Value"
                    ]
                ]
                .to_dict(
                    orient="records"
                )
        }