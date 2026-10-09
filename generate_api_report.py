import json
import os

from api.api_validation import (
    APIValidation
)

from api.main import (
    app,
    customer_service,
    churn_service,
    clv_service,
    recommendation_service,
    explainability_service
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(
        __file__
    )
)

ARTIFACT_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "api"
)

REPORT_PATH = os.path.join(
    ARTIFACT_DIRECTORY,
    "api_report.json"
)


# ============================================================
# IMPORTANT ARTIFACTS
# ============================================================

CUSTOMER_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "dashboard",
    "unified_customer_intelligence.csv"
)

CHURN_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "churn",
    "churn_model.pkl"
)

CLV_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "clv",
    "clv_model.pkl"
)

CLV_BANDS_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "clv",
    "clv_value_bands.json"
)

RECOMMENDER_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "recommendation",
    "recommender.pkl"
)

RECOMMENDATION_BATCH_PATH = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "recommendation",
    "predictions",
    "batch_recommendations.csv"
)

CHURN_GLOBAL_SHAP_PATH = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "explainability",
    "churn",
    "churn_global_shap_importance.csv"
)

CHURN_LOCAL_SHAP_PATH = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "explainability",
    "churn",
    "churn_dashboard_shap_values.csv"
)

CLV_GLOBAL_SHAP_PATH = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "explainability",
    "clv",
    "clv_global_shap_importance.csv"
)

CLV_LOCAL_SHAP_PATH = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "explainability",
    "clv",
    "clv_dashboard_shap_values.csv"
)


# ============================================================
# REQUIRED ROUTES
# ============================================================

REQUIRED_ROUTES = [

    (
        "GET",
        "/"
    ),

    (
        "GET",
        "/health"
    ),

    (
        "GET",
        "/data/status"
    ),

    (
        "GET",
        "/customers"
    ),

    (
        "GET",
        "/customers/{customer_id}"
    ),

    (
        "GET",
        "/customers/{customer_id}/summary"
    ),

    (
        "GET",
        "/models/churn/status"
    ),

    (
        "GET",
        "/models/clv/status"
    ),

    (
        "GET",
        "/models/recommendation/status"
    ),

    (
        "POST",
        "/predict/churn"
    ),

    (
        "POST",
        "/predict/clv"
    ),

    (
        "GET",
        "/recommend/{customer_id}"
    ),

    (
        "GET",
        "/explainability/status"
    ),

    (
        "GET",
        "/explain/global/churn"
    ),

    (
        "GET",
        "/explain/global/clv"
    ),

    (
        "GET",
        "/explain/churn/{customer_id}"
    ),

    (
        "GET",
        "/explain/clv/{customer_id}"
    ),

    (
        "GET",
        "/intelligence/{customer_id}"
    )
]


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        "========================================"
    )

    print(
        "FINAL FASTAPI VALIDATION"
    )

    print(
        "========================================"
        "\n"
    )

    os.makedirs(
        ARTIFACT_DIRECTORY,
        exist_ok=True
    )

    validator = (
        APIValidation()
    )

    # ========================================================
    # ARTIFACT VALIDATION
    # ========================================================

    artifact_paths = {

        "customer_dataset":
            CUSTOMER_DATA_PATH,

        "churn_model":
            CHURN_MODEL_PATH,

        "clv_model":
            CLV_MODEL_PATH,

        "clv_value_bands":
            CLV_BANDS_PATH,

        "recommendation_model":
            RECOMMENDER_PATH,

        "recommendation_batch":
            RECOMMENDATION_BATCH_PATH,

        "churn_global_shap":
            CHURN_GLOBAL_SHAP_PATH,

        "churn_local_shap":
            CHURN_LOCAL_SHAP_PATH,

        "clv_global_shap":
            CLV_GLOBAL_SHAP_PATH,

        "clv_local_shap":
            CLV_LOCAL_SHAP_PATH
    }

    artifact_validation = {}

    for name, path in artifact_paths.items():

        artifact_validation[
            name
        ] = (
            validator.check_file(
                path
            )
        )

    # ========================================================
    # SERVICE STATUS
    # ========================================================

    customer_status = (
        validator
        .safe_service_status(
            customer_service,
            "dataset_status"
        )
    )

    churn_status = (
        validator
        .safe_service_status(
            churn_service,
            "model_status"
        )
    )

    clv_status = (
        validator
        .safe_service_status(
            clv_service,
            "model_status"
        )
    )

    recommendation_status = (
        validator
        .safe_service_status(
            recommendation_service,
            "model_status"
        )
    )

    explainability_status = (
        validator
        .safe_service_status(
            explainability_service,
            "status"
        )
    )

    # ========================================================
    # ROUTE VALIDATION
    # ========================================================

    route_validation = (
        validator
        .validate_required_routes(
            app=
                app,

            required_routes=
                REQUIRED_ROUTES
        )
    )

    # ========================================================
    # OPENAPI
    # ========================================================

    openapi_validation = (
        validator
        .validate_openapi(
            app
        )
    )

    # ========================================================
    # CONDITIONS
    # ========================================================

    required_artifacts_exist = all(
        item[
            "exists"
        ]
        for item in artifact_validation.values()
    )

    service_checks_pass = all(
        [
            customer_status[
                "success"
            ],

            churn_status[
                "success"
            ],

            clv_status[
                "success"
            ],

            recommendation_status[
                "success"
            ],

            explainability_status[
                "success"
            ]
        ]
    )

    route_checks_pass = (
        route_validation[
            "all_present"
        ]
    )

    openapi_pass = (
        openapi_validation[
            "valid"
        ]
    )

    overall_status = (
        validator.overall_status(
            [
                required_artifacts_exist,
                service_checks_pass,
                route_checks_pass,
                openapi_pass
            ]
        )
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    report = {

        "phase":
            "Phase 7 — FastAPI",

        "status":
            overall_status,

        "api": {

            "title":
                app.title,

            "version":
                app.version,

            "docs_path":
                "/docs",

            "openapi_path":
                "/openapi.json"
        },

        "artifact_validation":
            artifact_validation,

        "service_status": {

            "customer_service":
                customer_status,

            "churn_service":
                churn_status,

            "clv_service":
                clv_status,

            "recommendation_service":
                recommendation_status,

            "explainability_service":
                explainability_status
        },

        "route_validation":
            route_validation,

        "registered_routes":
            validator.registered_routes(
                app
            ),

        "openapi_validation":
            openapi_validation,

        "capabilities": {

            "customer_lookup":
                True,

            "customer_pagination":
                True,

            "live_churn_prediction":
                True,

            "live_clv_prediction":
                True,

            "recommendation_lookup":
                True,

            "global_shap_explainability":
                True,

            "customer_shap_explainability":
                True,

            "unified_customer_intelligence":
                True,

            "swagger_documentation":
                True
        },

        "important_semantics": {

            "customer_id_column":
                "Customer ID",

            "clv_target":
                "Future 90-day revenue",

            "clv_double_inverse_transform":
                False,

            "recommendation_score":
                (
                    "Ranking signal, not calibrated "
                    "purchase probability."
                ),

            "shap_interpretation":
                (
                    "Model contribution explanation, "
                    "not causal inference."
                ),

            "business_priority":
                (
                    "Rule-based decision-support layer, "
                    "not an additional ML model."
                )
        },

        "known_limitations": [

            (
                "Customer-level SHAP explanations are "
                "currently available only for the "
                "precomputed explanation sample."
            ),

            (
                "Cold-start recommendation capability "
                "depends on the selected persisted "
                "recommender and is not universally assumed."
            ),

            (
                "The current development CORS configuration "
                "allows all origins and should be restricted "
                "for a real production deployment."
            ),

            (
                "Authentication and authorization are not "
                "implemented in this portfolio API."
            ),

            (
                "The API currently runs as a single service "
                "without an external production datastore."
            )
        ],

        "validation_summary": {

            "required_artifacts_exist":
                required_artifacts_exist,

            "services_respond":
                service_checks_pass,

            "required_routes_registered":
                route_checks_pass,

            "openapi_valid":
                openapi_pass,

            "overall_status":
                overall_status
        }
    }

    # ========================================================
    # SAVE
    # ========================================================

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    print(
        f"Overall Status: {overall_status}"
    )

    print(
        "\nArtifact Validation:"
    )

    for name, result in artifact_validation.items():

        status = (
            "OK"
            if result[
                "exists"
            ]
            else
            "MISSING"
        )

        print(
            f"- {name}: {status}"
        )

    print(
        "\nService Validation:"
    )

    print(
        (
            "- Customer Service: "
            f"{customer_status['success']}"
        )
    )

    print(
        (
            "- Churn Service: "
            f"{churn_status['success']}"
        )
    )

    print(
        (
            "- CLV Service: "
            f"{clv_status['success']}"
        )
    )

    print(
        (
            "- Recommendation Service: "
            f"{recommendation_status['success']}"
        )
    )

    print(
        (
            "- Explainability Service: "
            f"{explainability_status['success']}"
        )
    )

    print(
        "\nRoutes:"
    )

    for route, exists in (
        route_validation[
            "routes"
        ]
        .items()
    ):

        print(
            f"- {route}: "
            f"{'OK' if exists else 'MISSING'}"
        )

    print(
        "\nOpenAPI:"
    )

    print(
        (
            f"- Valid: "
            f"{openapi_validation['valid']}"
        )
    )

    print(
        (
            f"- Registered paths: "
            f"{openapi_validation['path_count']}"
        )
    )

    print(
        "\nReport saved:"
    )

    print(
        REPORT_PATH
    )

    if overall_status == "PASS":

        print(
            "\nPHASE 7 VALIDATION PASSED."
        )

    else:

        print(
            "\nSome API components require review."
        )


if __name__ == "__main__":

    main()