import sys

import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger
from src.utils.model_persistence import ModelPersistence


class ChurnPredictor:

    def __init__(
        self,
        model_path: str = "models/churn/churn_model.pkl"
    ):

        try:

            logger.info(
                "Initializing churn predictor."
            )

            self.features = [
                "Recency",
                "Frequency",
                "Monetary",
                "TotalItems",
                "AverageOrderValue",
                "Tenure"
            ]

            self.model_path = model_path

            persistence = ModelPersistence()

            self.model = persistence.load_model(
                self.model_path
            )

            logger.info(
                "Churn predictor initialized successfully."
            )

        except Exception as e:

            logger.error(
                "Churn predictor initialization failed."
            )

            raise CustomException(
                e,
                sys
            )

    def validate_input(
        self,
        customer_data: dict
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Validating churn prediction input."
            )

            missing_features = [
                feature
                for feature in self.features
                if feature not in customer_data
            ]

            if missing_features:

                raise ValueError(
                    f"Missing required features: "
                    f"{missing_features}"
                )

            input_df = pd.DataFrame(
                [
                    {
                        feature: customer_data[feature]
                        for feature in self.features
                    }
                ]
            )

            for feature in self.features:

                input_df[feature] = pd.to_numeric(
                    input_df[feature],
                    errors="coerce"
                )

            if input_df.isnull().any().any():

                raise ValueError(
                    "Customer features contain "
                    "missing or non-numeric values."
                )

            if (
                input_df[self.features] < 0
            ).any().any():

                raise ValueError(
                    "Customer features cannot "
                    "contain negative values."
                )

            logger.info(
                "Prediction input validated successfully."
            )

            return input_df

        except Exception as e:

            logger.error(
                "Prediction input validation failed."
            )

            raise CustomException(
                e,
                sys
            )

    def get_risk_level(
        self,
        probability: float
    ) -> str:

        try:

            if probability < 0.30:

                return "Low"

            elif probability < 0.70:

                return "Medium"

            else:

                return "High"

        except Exception as e:

            logger.error(
                "Risk level calculation failed."
            )

            raise CustomException(
                e,
                sys
            )

    def predict(
        self,
        customer_data: dict
    ) -> dict:

        try:

            logger.info(
                "Starting churn prediction."
            )

            input_df = self.validate_input(
                customer_data
            )

            prediction = int(
                self.model.predict(
                    input_df
                )[0]
            )

            churn_probability = None

            if hasattr(
                self.model,
                "predict_proba"
            ):

                probabilities = (
                    self.model.predict_proba(
                        input_df
                    )
                )

                churn_probability = float(
                    probabilities[0][1]
                )

            risk_level = None

            if churn_probability is not None:

                risk_level = (
                    self.get_risk_level(
                        churn_probability
                    )
                )

            result = {
                "prediction": prediction,
                "churn_probability":
                    churn_probability,
                "risk_level":
                    risk_level
            }

            logger.info(
                f"Churn prediction completed: "
                f"{result}"
            )

            return result

        except Exception as e:

            logger.error(
                "Churn prediction failed."
            )

            raise CustomException(
                e,
                sys
            )