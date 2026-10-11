from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)


# ============================================================
# REQUIRED DIRECTORIES
# ============================================================

REQUIRED_DIRECTORIES = [
    PROJECT_ROOT / "api",
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "artifacts",
]


# ============================================================
# REQUIRED SOURCE FILES
# ============================================================

REQUIRED_SOURCE_FILES = [
    PROJECT_ROOT
    / "api"
    / "main.py",

    PROJECT_ROOT
    / "api"
    / "schemas.py",

    PROJECT_ROOT
    / "api"
    / "customer_service.py",

    PROJECT_ROOT
    / "api"
    / "churn_service.py",

    PROJECT_ROOT
    / "api"
    / "clv_service.py",

    PROJECT_ROOT
    / "api"
    / "recommendation_service.py",

    PROJECT_ROOT
    / "api"
    / "explainability_service.py",

    PROJECT_ROOT
    / "api"
    / "intelligence_service.py",

    PROJECT_ROOT
    / "Dockerfile",

    PROJECT_ROOT
    / "requirements.txt",
]


# ============================================================
# IMPORTANT PRODUCTION ARTIFACTS
# ============================================================

PRODUCTION_ARTIFACTS = {
    "churn_model":
        PROJECT_ROOT
        / "models"
        / "churn"
        / "churn_model.pkl",

    "clv_model":
        PROJECT_ROOT
        / "models"
        / "clv"
        / "clv_model.pkl",

    "clv_value_bands":
        PROJECT_ROOT
        / "models"
        / "clv"
        / "clv_value_bands.json",

    "recommendation_model":
        PROJECT_ROOT
        / "models"
        / "recommendation"
        / "recommender.pkl",

    "recommendation_metadata":
        PROJECT_ROOT
        / "models"
        / "recommendation"
        / "recommender_metadata.json",

    "unified_customer_intelligence":
        PROJECT_ROOT
        / "artifacts"
        / "dashboard"
        / "unified_customer_intelligence.csv",

    "recommendation_batch":
        PROJECT_ROOT
        / "artifacts"
        / "recommendation"
        / "predictions"
        / "batch_recommendations.csv",
}


# ============================================================
# REQUIRED API ROUTES
# ============================================================

REQUIRED_ROUTES = {
    "/",
    "/health",
    "/data/status",
    "/customers",
    "/customers/{customer_id}",
    "/customers/{customer_id}/summary",
    "/models/churn/status",
    "/models/clv/status",
    "/models/recommendation/status",
    "/predict/churn",
    "/predict/clv",
    "/recommend/{customer_id}",
    "/explainability/status",
    "/explain/global/churn",
    "/explain/global/clv",
    "/explain/churn/{customer_id}",
    "/explain/clv/{customer_id}",
    "/intelligence/{customer_id}",
}


# ============================================================
# HELPER
# ============================================================

def status_result(
    success: bool,
    detail: str,
) -> dict[str, Any]:

    return {
        "success":
            success,

        "detail":
            detail,
    }


# ============================================================
# DIRECTORY VALIDATION
# ============================================================

def validate_directories(
) -> dict[str, Any]:

    results = {}

    for directory in (
        REQUIRED_DIRECTORIES
    ):

        relative_path = (
            directory.relative_to(
                PROJECT_ROOT
            )
        )

        exists = (
            directory.exists()
            and
            directory.is_dir()
        )

        results[
            str(relative_path)
        ] = exists

    success = all(
        results.values()
    )

    return {
        "success":
            success,

        "directories":
            results,
    }


# ============================================================
# SOURCE FILE VALIDATION
# ============================================================

def validate_source_files(
) -> dict[str, Any]:

    results = {}

    for file_path in (
        REQUIRED_SOURCE_FILES
    ):

        relative_path = (
            file_path.relative_to(
                PROJECT_ROOT
            )
        )

        exists = (
            file_path.exists()
            and
            file_path.is_file()
        )

        results[
            str(relative_path)
        ] = exists

    success = all(
        results.values()
    )

    return {
        "success":
            success,

        "files":
            results,
    }


# ============================================================
# PRODUCTION ARTIFACT VALIDATION
# ============================================================

def validate_production_artifacts(
) -> dict[str, Any]:

    results = {}

    for name, artifact_path in (
        PRODUCTION_ARTIFACTS.items()
    ):

        results[name] = {
            "exists":
                artifact_path.exists(),

            "path":
                str(
                    artifact_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
        }

    success = all(
        item["exists"]
        for item in results.values()
    )

    return {
        "success":
            success,

        "artifacts":
            results,
    }


# ============================================================
# IMPORT VALIDATION
# ============================================================

def validate_imports(
) -> dict[str, Any]:

    modules = [
        "api.main",
        "api.schemas",
        "api.customer_service",
        "api.churn_service",
        "api.clv_service",
        "api.recommendation_service",
        "api.explainability_service",
        "api.intelligence_service",
    ]

    results = {}

    for module_name in modules:

        try:

            importlib.import_module(
                module_name
            )

            results[
                module_name
            ] = {
                "success":
                    True,

                "error":
                    None,
            }

        except Exception as error:

            results[
                module_name
            ] = {
                "success":
                    False,

                "error":
                    str(error),
            }

    success = all(
        result["success"]
        for result in results.values()
    )

    return {
        "success":
            success,

        "modules":
            results,
    }


# ============================================================
# FASTAPI APPLICATION VALIDATION
# ============================================================

def validate_fastapi_app(
) -> dict[str, Any]:

    try:

        from api.main import app

    except Exception as error:

        return {
            "success":
                False,

            "error":
                (
                    "Could not import FastAPI app: "
                    f"{error}"
                ),

            "routes":
                [],
        }

    if app is None:

        return {
            "success":
                False,

            "error":
                "FastAPI app is None.",

            "routes":
                [],
        }

    registered_routes = {
        route.path
        for route in app.routes
    }

    missing_routes = sorted(
        REQUIRED_ROUTES
        -
        registered_routes
    )

    success = (
        len(
            missing_routes
        )
        ==
        0
    )

    return {
        "success":
            success,

        "registered_route_count":
            len(
                registered_routes
            ),

        "registered_routes":
            sorted(
                registered_routes
            ),

        "missing_routes":
            missing_routes,
    }


# ============================================================
# OPENAPI VALIDATION
# ============================================================

def validate_openapi(
) -> dict[str, Any]:

    try:

        from api.main import app

        schema = app.openapi()

    except Exception as error:

        return status_result(
            False,
            (
                "OpenAPI generation failed: "
                f"{error}"
            ),
        )

    if not isinstance(
        schema,
        dict,
    ):

        return status_result(
            False,
            "OpenAPI schema is not a dictionary.",
        )

    required_keys = {
        "openapi",
        "info",
        "paths",
    }

    missing = (
        required_keys
        -
        set(
            schema.keys()
        )
    )

    if missing:

        return status_result(
            False,
            (
                "OpenAPI schema missing keys: "
                f"{sorted(missing)}"
            ),
        )

    if not schema[
        "paths"
    ]:

        return status_result(
            False,
            "OpenAPI schema contains no paths.",
        )

    return status_result(
        True,
        (
            "OpenAPI schema generated "
            "successfully."
        ),
    )


# ============================================================
# MLFLOW MODULE VALIDATION
# ============================================================

def validate_mlflow_modules(
) -> dict[str, Any]:

    modules = [
        "src.mlflow_tracking.config",
        "src.mlflow_tracking.tracker",
        "src.mlflow_tracking.churn_tracking",
        "src.mlflow_tracking.clv_tracking",
        "src.mlflow_tracking.recommendation_tracking",
        "src.mlflow_tracking.mlflow_validation",
    ]

    results = {}

    for module_name in modules:

        try:

            importlib.import_module(
                module_name
            )

            results[
                module_name
            ] = {
                "success":
                    True,

                "error":
                    None,
            }

        except Exception as error:

            results[
                module_name
            ] = {
                "success":
                    False,

                "error":
                    str(error),
            }

    success = all(
        item["success"]
        for item in results.values()
    )

    return {
        "success":
            success,

        "modules":
            results,
    }


# ============================================================
# DOCKERFILE VALIDATION
# ============================================================

def validate_dockerfile(
) -> dict[str, Any]:

    dockerfile = (
        PROJECT_ROOT
        / "Dockerfile"
    )

    if not dockerfile.exists():

        return status_result(
            False,
            "Dockerfile not found.",
        )

    content = dockerfile.read_text(
        encoding="utf-8"
    )

    required_tokens = [
        "FROM python:",
        "WORKDIR",
        "EXPOSE 8000",
        "uvicorn",
        "api.main:app",
    ]

    missing_tokens = [
        token
        for token in required_tokens
        if token not in content
    ]

    if missing_tokens:

        return {
            "success":
                False,

            "detail":
                (
                    "Dockerfile missing required "
                    "configuration."
                ),

            "missing":
                missing_tokens,
        }

    return {
        "success":
            True,

        "detail":
            "Dockerfile structure looks valid.",

        "missing":
            [],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        "========================================"
    )

    print(
        "CI PROJECT VALIDATION"
    )

    print(
        "========================================"
        "\n"
    )

    validation = {
        "directories":
            validate_directories(),

        "source_files":
            validate_source_files(),

        "production_artifacts":
            validate_production_artifacts(),

        "imports":
            validate_imports(),

        "fastapi":
            validate_fastapi_app(),

        "openapi":
            validate_openapi(),

        "mlflow":
            validate_mlflow_modules(),

        "dockerfile":
            validate_dockerfile(),
    }

    overall_success = all(
        result[
            "success"
        ]
        for result in validation.values()
    )

    validation[
        "overall_status"
    ] = (
        "PASS"
        if overall_success
        else
        "FAIL"
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    for name, result in (
        validation.items()
    ):

        if name == "overall_status":

            continue

        print(
            f"{name}: "
            f"{'PASS' if result['success'] else 'FAIL'}"
        )

    print(
        "\nOverall Status:"
    )

    print(
        validation[
            "overall_status"
        ]
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    output_directory = (
        PROJECT_ROOT
        / "artifacts"
        / "ci"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        output_directory
        / "ci_validation_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            validation,
            file,
            indent=4,
        )

    print(
        "\nReport saved:"
    )

    print(
        report_path
    )

    # --------------------------------------------------------
    # Correct CI exit code
    # --------------------------------------------------------

    if not overall_success:

        print(
            "\nCI validation failed."
        )

        sys.exit(
            1
        )

    print(
        "\nCI validation passed."
    )

    sys.exit(
        0
    )


if __name__ == "__main__":

    main()