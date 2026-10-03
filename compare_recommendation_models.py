import json
import os

import pandas as pd


def load_metrics(
    file_path: str,
    model_name: str
):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(
            file
        )

    rows = []

    for k in [
        "5",
        "10",
        "20"
    ]:

        metrics = (
            data[
                "metrics"
            ][
                k
            ]
        )

        rows.append(
            {
                "Model":
                    model_name,

                "K":
                    int(
                        k
                    ),

                "PrecisionAtK":
                    metrics[
                        "precision_at_k"
                    ],

                "RecallAtK":
                    metrics[
                        "recall_at_k"
                    ],

                "HitRateAtK":
                    metrics[
                        "hit_rate_at_k"
                    ],

                "CatalogCoverage":
                    metrics[
                        "catalog_coverage"
                    ],

                "UniqueRecommendedProducts":
                    metrics[
                        "unique_recommended_products"
                    ]
            }
        )

    return rows


def main():

    # =================================
    # PATHS
    # =================================

    popularity_path = (
        "artifacts/recommendation/"
        "baseline/"
        "popularity_metrics.json"
    )

    item_cf_path = (
        "artifacts/recommendation/"
        "item_cf/"
        "item_cf_metrics.json"
    )

    matrix_factorization_path = (
        "artifacts/recommendation/"
        "matrix_factorization/"
        "matrix_factorization_metrics.json"
    )

    output_path = (
        "artifacts/recommendation/"
        "recommendation_model_comparison.csv"
    )

    required_files = [
        popularity_path,
        item_cf_path,
        matrix_factorization_path
    ]

    for file_path in required_files:

        if not os.path.exists(
            file_path
        ):

            raise FileNotFoundError(
                f"Recommendation metric file "
                f"not found: {file_path}"
            )

    # =================================
    # BUILD COMPARISON
    # =================================

    rows = []

    rows.extend(
        load_metrics(
            popularity_path,
            "Popularity Baseline"
        )
    )

    rows.extend(
        load_metrics(
            item_cf_path,
            "Item-Based Collaborative Filtering"
        )
    )

    rows.extend(
        load_metrics(
            matrix_factorization_path,
            "Matrix Factorization"
        )
    )

    comparison = pd.DataFrame(
        rows
    )

    comparison = (
        comparison
        .sort_values(
            by=[
                "K",
                "RecallAtK",
                "HitRateAtK"
            ],
            ascending=[
                True,
                False,
                False
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # =================================
    # SAVE
    # =================================

    comparison.to_csv(
        output_path,
        index=False
    )

    # =================================
    # OUTPUT
    # =================================

    print(
        "\n================================"
    )

    print(
        "RECOMMENDATION MODEL COMPARISON"
    )

    print(
        "================================"
    )

    for k in [
        5,
        10,
        20
    ]:

        print(
            f"\n---------- K={k} ----------"
        )

        k_results = (
            comparison[
                comparison[
                    "K"
                ]
                ==
                k
            ]
        )

        print(
            k_results.to_string(
                index=False
            )
        )

    print(
        f"\nComparison saved to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":

    main()