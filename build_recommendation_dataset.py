import os

import pandas as pd

from src.preprocessing.recommendation_cleaning import (
    RecommendationDataCleaning
)

from src.feature_engineering.recommendation_interactions import (
    RecommendationInteractionBuilder
)

from src.feature_engineering.product_catalog import (
    ProductCatalogBuilder
)


def main():

    # =================================
    # PATHS
    # =================================

    input_path = (
        "data/processed/"
        "retail.csv"
    )

    output_directory = (
        "artifacts/recommendation"
    )

    interaction_path = os.path.join(
        output_directory,
        "customer_product_interactions.csv"
    )

    catalog_path = os.path.join(
        output_directory,
        "product_catalog.csv"
    )

    cleaned_transaction_path = os.path.join(
        output_directory,
        "clean_purchase_transactions.csv"
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # =================================
    # LOAD DATA
    # =================================

    df = pd.read_csv(
        input_path
    )

    print(
        "\nRaw transaction shape:"
    )

    print(
        df.shape
    )

    # =================================
    # CLEAN
    # =================================

    cleaner = (
        RecommendationDataCleaning()
    )

    clean_df = (
        cleaner.clean_purchase_transactions(
            df
        )
    )

    # =================================
    # BUILD INTERACTIONS
    # =================================

    interaction_builder = (
        RecommendationInteractionBuilder()
    )

    interactions = (
        interaction_builder.build_interactions(
            clean_df
        )
    )

    # =================================
    # BUILD PRODUCT CATALOG
    # =================================

    catalog_builder = (
        ProductCatalogBuilder()
    )

    catalog = (
        catalog_builder.build_catalog(
            clean_df
        )
    )

    # =================================
    # SAVE
    # =================================

    clean_df.to_csv(
        cleaned_transaction_path,
        index=False
    )

    interactions.to_csv(
        interaction_path,
        index=False
    )

    catalog.to_csv(
        catalog_path,
        index=False
    )

    # =================================
    # BASIC STATISTICS
    # =================================

    unique_customers = (
        interactions[
            "CustomerID"
        ]
        .nunique()
    )

    unique_products = (
        interactions[
            "StockCode"
        ]
        .nunique()
    )

    total_interactions = (
        len(
            interactions
        )
    )

    # Customer-product matrix density
    possible_interactions = (
        unique_customers
        *
        unique_products
    )

    if possible_interactions > 0:

        density = (
            total_interactions
            /
            possible_interactions
            *
            100
        )

    else:

        density = 0

    # =================================
    # OUTPUT
    # =================================

    print(
        "\n================================"
    )

    print(
        "RECOMMENDATION DATASET"
    )

    print(
        "================================"
    )

    print(
        f"\nClean transactions: "
        f"{len(clean_df)}"
    )

    print(
        f"Unique customers: "
        f"{unique_customers}"
    )

    print(
        f"Unique products: "
        f"{unique_products}"
    )

    print(
        f"Customer-product interactions: "
        f"{total_interactions}"
    )

    print(
        f"Interaction matrix density: "
        f"{density:.6f}%"
    )

    print(
        "\nInteraction columns:"
    )

    print(
        interactions.columns.tolist()
    )

    print(
        "\nSample interactions:"
    )

    print(
        interactions.head(
            10
        ).to_string(
            index=False
        )
    )

    print(
        f"\nInteractions saved to:"
        f"\n{interaction_path}"
    )

    print(
        f"\nProduct catalog saved to:"
        f"\n{catalog_path}"
    )


if __name__ == "__main__":

    main()