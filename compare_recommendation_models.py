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
    # METRIC FILES
    # =================================

    metric_files = [

        (
            "Popularity Baseline",

            "artifacts/recommendation/"
            "baseline/"
            "popularity_metrics.json"
        ),

        (
            "Item-Based Collaborative Filtering",

            "artifacts/recommendation/"
            "item_cf/"
            "item_cf_metrics.json"
        ),

        (
            "Matrix Factorization",

            "artifacts/recommendation/"
            "matrix_factorization/"
            "matrix_factorization_metrics.json"
        ),

        (
            "Hybrid Recommender",

            "artifacts/recommendation/"
            "hybrid/"
            "hybrid_metrics.json"
        )
    ]

    output_path = (
        "artifacts/recommendation/"
        "recommendation_model_comparison.csv"
    )

    rows = []

    for (
        model_name,
        file_path
    ) in metric_files:

        if not os.path.exists(
            file_path
        ):

            raise FileNotFoundError(
                f"Recommendation metric file "
                f"not found: {file_path}"
            )

        rows.extend(
            load_metrics(
                file_path=
                    file_path,

                model_name=
                    model_name
            )
        )

    # =================================
    # DATAFRAME
    # =================================

    comparison = pd.DataFrame(
        rows
    )

    comparison = (
        comparison
        .sort_values(
            by=[
                "K",
                "RecallAtK",
                "HitRateAtK",
                "PrecisionAtK"
            ],
            ascending=[
                True,
                False,
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
    # DISPLAY
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

        result = (
            comparison[
                comparison[
                    "K"
                ]
                ==
                k
            ]
        )

        print(
            result.to_string(
                index=False
            )
        )

    print(
        f"\nComparison saved to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":

    main()