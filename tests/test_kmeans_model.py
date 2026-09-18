import pandas as pd

from sklearn.cluster import KMeans

from src.training.kmeans_model import (
    CustomerSegmentationModel
)


def test_kmeans_model():

    X = pd.DataFrame(
        {
            "Recency": [
                -1.1,
                -1.0,
                -0.9,
                0.9,
                1.0,
                1.1,
                2.8,
                2.9,
                3.0
            ],

            "Frequency": [
                1.1,
                1.0,
                0.9,
                -0.9,
                -1.0,
                -1.1,
                2.8,
                2.9,
                3.0
            ],

            "Monetary": [
                1.0,
                1.1,
                0.9,
                -1.0,
                -0.9,
                -1.1,
                2.9,
                3.0,
                2.8
            ],

            "TotalItems": [
                1.1,
                1.0,
                0.9,
                -0.9,
                -1.0,
                -1.1,
                3.0,
                2.8,
                2.9
            ],

            "AverageOrderValue": [
                1.0,
                0.9,
                1.1,
                -1.0,
                -1.1,
                -0.9,
                2.8,
                3.0,
                2.9
            ],

            "Tenure": [
                1.1,
                1.0,
                0.9,
                -0.9,
                -1.0,
                -1.1,
                2.9,
                2.8,
                3.0
            ]
        }
    )

    trainer = (
        CustomerSegmentationModel(
            n_clusters=3,
            random_state=42
        )
    )

    model = trainer.train(
        X
    )

    assert model is not None

    assert isinstance(
        model,
        KMeans
    )

    assert (
        model.n_clusters
        == 3
    )

    clusters = (
        trainer.assign_clusters(
            model,
            X
        )
    )

    assert clusters is not None

    assert (
        len(clusters)
        ==
        len(X)
    )

    assert (
        clusters.nunique()
        <= 3
    )

    assert (
        clusters.min()
        >= 0
    )

    assert (
        clusters.max()
        < 3
    )
    