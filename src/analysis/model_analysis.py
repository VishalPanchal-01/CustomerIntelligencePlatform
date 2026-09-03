import sys

import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger

class ModelAnalysis:
    def analyze_logistic_coefficients(self,model,feature_names) -> pd.DataFrame:
        try:
            logger.info("Analyzing logistic regression coefficients.")
            classifier = (model.named_steps["classifier"])

            coefficients = (classifier.coef_[0])
            report = pd.DataFrame({
                    "Feature": feature_names,
                    "Coefficient": coefficients
                })

            report["AbsoluteCoefficient"] = (report["Coefficient"].abs())

            report = (report.sort_values(by="AbsoluteCoefficient",ascending=False).reset_index(drop=True))

            logger.info(f"Logistic coefficient report:\n"f"{report}")

            return report

        except Exception as e:
            logger.error("Logistic coefficient analysis failed.")
            raise CustomException(e,sys)