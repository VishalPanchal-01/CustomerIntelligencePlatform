import numpy as np
import pandas as pd


class SHAPUtils:

    # =========================================================
    # FEATURE VALIDATION
    # =========================================================

    @staticmethod
    def validate_features(
        df: pd.DataFrame,
        required_features: list
    ) -> pd.DataFrame:

        missing = [
            feature
            for feature in required_features
            if feature not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Missing required features: {missing}"
            )

        result = (
            df[
                required_features
            ]
            .copy()
        )

        for column in required_features:

            result[column] = pd.to_numeric(
                result[column],
                errors="coerce"
            )

        if result.isnull().any().any():

            null_columns = (
                result
                .columns[
                    result.isnull().any()
                ]
                .tolist()
            )

            raise ValueError(
                "Missing or non-numeric values found in "
                f"features: {null_columns}"
            )

        return result


    # =========================================================
    # SAMPLE BACKGROUND DATA
    # =========================================================

    @staticmethod
    def sample_background(
        X: pd.DataFrame,
        max_samples: int = 200,
        random_state: int = 42
    ) -> pd.DataFrame:

        if len(X) <= max_samples:

            return (
                X
                .copy()
                .reset_index(
                    drop=True
                )
            )

        return (
            X
            .sample(
                n=max_samples,
                random_state=random_state
            )
            .reset_index(
                drop=True
            )
        )


    # =========================================================
    # SAMPLE EXPLANATION DATA
    # =========================================================

    @staticmethod
    def sample_explanation_rows(
        X: pd.DataFrame,
        max_samples: int = 500,
        random_state: int = 42
    ) -> pd.DataFrame:

        if len(X) <= max_samples:

            return (
                X
                .copy()
                .reset_index(
                    drop=True
                )
            )

        return (
            X
            .sample(
                n=max_samples,
                random_state=random_state
            )
            .reset_index(
                drop=True
            )
        )


    # =========================================================
    # NORMALIZE SHAP VALUES
    # =========================================================

    @staticmethod
    def extract_binary_class_values(
        shap_values
    ) -> np.ndarray:

        values = (
            shap_values.values
            if hasattr(
                shap_values,
                "values"
            )
            else shap_values
        )

        values = np.asarray(
            values
        )

        # -----------------------------------------------------
        # Common binary-classification formats:
        #
        # (rows, features)
        #
        # or
        #
        # (rows, features, classes)
        # -----------------------------------------------------

        if values.ndim == 2:

            return values

        if (
            values.ndim == 3
            and
            values.shape[-1] >= 2
        ):

            return values[
                :,
                :,
                1
            ]

        raise ValueError(
            "Unsupported SHAP value shape: "
            f"{values.shape}"
        )


    # =========================================================
    # EXTRACT BASE VALUE
    # =========================================================

    @staticmethod
    def extract_binary_base_value(
        shap_values,
        row_index: int = 0
    ) -> float:

        if not hasattr(
            shap_values,
            "base_values"
        ):

            return 0.0

        base_values = np.asarray(
            shap_values.base_values
        )

        if base_values.ndim == 0:

            return float(
                base_values
            )

        if base_values.ndim == 1:

            return float(
                base_values[
                    row_index
                ]
            )

        if (
            base_values.ndim == 2
            and
            base_values.shape[
                1
            ]
            >=
            2
        ):

            return float(
                base_values[
                    row_index,
                    1
                ]
            )

        return float(
            np.ravel(
                base_values
            )[0]
        )