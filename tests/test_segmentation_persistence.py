import os

from sklearn.cluster import KMeans

from src.utils.segmentation_persistence import (
    SegmentationPersistence
)


def test_segmentation_persistence(
    tmp_path
):

    model = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )

    model_path = os.path.join(
        tmp_path,
        "kmeans_test.pkl"
    )

    persistence = (
        SegmentationPersistence()
    )

    saved_path = (
        persistence.save_artifact(
            model,
            model_path
        )
    )

    assert os.path.exists(
        saved_path
    )

    loaded_model = (
        persistence.load_artifact(
            saved_path
        )
    )

    assert loaded_model is not None

    assert isinstance(
        loaded_model,
        KMeans
    )

    assert (
        loaded_model.n_clusters
        == 2
    )