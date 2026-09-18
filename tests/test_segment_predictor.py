import os

import pandas as pd

from src.preprocessing.segmentation_preprocessing import (
    SegmentationPreprocessor
)

from src.training.kmeans_model import (
    CustomerSegmentationModel
)

from src.prediction.segment_predictor import (
    SegmentPredictor
)

from src.utils.segmentation_persistence import (
    SegmentationPersistence
)


def test_segment_predictor(
    tmp_path
):

    # ---------------------------------
    # Create training data
    # ---------------------------------

    df = pd.DataFrame(
        {
            "CustomerID": [
                101,
                102,
                103,
                104,
                105,
                106
            ],

            "Recency": [
                10,
                20,
                30,
                120,
                150,
                180
            ],

            "Frequency": [
                10,
                8,
                7,
                2,
                1,
                1
            ],

            "Monetary": [
                5000,
                4000,
                3500,
                500,
                200,
                100
            ],

            "TotalItems": [
                500,
                400,
                350,
                50,
                20,
                10
            ],

            "AverageOrderValue": [
                500,
                500,
                500,
                250,
                200,
                100
            ],

            "Tenure": [
                400,
                350,
                300,
                100,
                50,
                20
            ]
        }
    )

    # ---------------------------------
    # Fit preprocessor
    # ---------------------------------

    preprocessor = (
        SegmentationPreprocessor()
    )

    (
        customer_ids,
        scaled_features
    ) = preprocessor.prepare_features(
        df
    )

    # ---------------------------------
    # Train K-Means
    # ---------------------------------

    trainer = (
        CustomerSegmentationModel(
            n_clusters=2,
            random_state=42
        )
    )

    model = trainer.train(
        scaled_features
    )

    # ---------------------------------
    # Create temporary artifact paths
    # ---------------------------------

    preprocessor_path = os.path.join(
        tmp_path,
        "preprocessor.pkl"
    )

    model_path = os.path.join(
        tmp_path,
        "kmeans.pkl"
    )

    mapping_path = os.path.join(
        tmp_path,
        "segment_summary.csv"
    )

    # ---------------------------------
    # Save trained artifacts
    # ---------------------------------

    persistence = (
        SegmentationPersistence()
    )

    persistence.save_artifact(
        preprocessor,
        preprocessor_path
    )

    persistence.save_artifact(
        model,
        model_path
    )

    # ---------------------------------
    # Create segment mapping
    # ---------------------------------

    mapping_df = pd.DataFrame(
        {
            "Cluster": [
                0,
                1
            ],

            "SegmentName": [
                "Segment A",
                "Segment B"
            ],

            "Recommendation": [
                "Recommendation A",
                "Recommendation B"
            ]
        }
    )

    mapping_df.to_csv(
        mapping_path,
        index=False
    )

    # ---------------------------------
    # Initialize predictor
    # ---------------------------------

    predictor = (
        SegmentPredictor(
            preprocessor_path=
                preprocessor_path,

            model_path=
                model_path,

            mapping_path=
                mapping_path
        )
    )

    # ---------------------------------
    # New customer
    # ---------------------------------

    customer_data = {
        "Recency": 15,
        "Frequency": 9,
        "Monetary": 4500,
        "TotalItems": 450,
        "AverageOrderValue": 500,
        "Tenure": 375
    }

    result = (
        predictor.predict_segment(
            customer_data
        )
    )

    # ---------------------------------
    # Assertions
    # ---------------------------------

    assert result is not None

    assert (
        "cluster"
        in result
    )

    assert (
        "segment_name"
        in result
    )

    assert (
        "recommendation"
        in result
    )

    assert (
        result["cluster"]
        in [0, 1]
    )

    assert (
        result[
            "segment_name"
        ]
        in [
            "Segment A",
            "Segment B"
        ]
    )