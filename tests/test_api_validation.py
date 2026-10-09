import os

from fastapi import FastAPI

from api.api_validation import (
    APIValidation
)


# ============================================================
# TEST APP
# ============================================================

def create_test_app():

    app = FastAPI(
        title=
            "Test API",

        version=
            "1.0"
    )

    @app.get(
        "/health"
    )
    def health():

        return {
            "status":
                "healthy"
        }

    @app.post(
        "/predict"
    )
    def predict():

        return {
            "prediction":
                1
        }

    return app


# ============================================================
# FILE CHECK
# ============================================================

def test_check_file(
    tmp_path
):

    path = os.path.join(
        tmp_path,
        "artifact.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "test"
        )

    result = (
        APIValidation
        .check_file(
            path
        )
    )

    assert (
        result[
            "exists"
        ]
        is True
    )

    assert (
        result[
            "size_bytes"
        ]
        >
        0
    )


# ============================================================
# REGISTERED ROUTES
# ============================================================

def test_registered_routes():

    app = create_test_app()

    routes = (
        APIValidation
        .registered_routes(
            app
        )
    )

    paths = [
        item[
            "path"
        ]
        for item in routes
    ]

    assert (
        "/health"
        in paths
    )

    assert (
        "/predict"
        in paths
    )


# ============================================================
# ROUTE EXISTS
# ============================================================

def test_route_exists():

    app = create_test_app()

    assert (
        APIValidation
        .route_exists(
            app,
            "/health",
            "GET"
        )
        is True
    )

    assert (
        APIValidation
        .route_exists(
            app,
            "/predict",
            "POST"
        )
        is True
    )

    assert (
        APIValidation
        .route_exists(
            app,
            "/unknown",
            "GET"
        )
        is False
    )


# ============================================================
# REQUIRED ROUTES
# ============================================================

def test_required_routes():

    app = create_test_app()

    validator = (
        APIValidation()
    )

    result = (
        validator
        .validate_required_routes(
            app,
            [
                (
                    "GET",
                    "/health"
                ),

                (
                    "POST",
                    "/predict"
                )
            ]
        )
    )

    assert (
        result[
            "all_present"
        ]
        is True
    )


# ============================================================
# OPENAPI
# ============================================================

def test_openapi_validation():

    app = create_test_app()

    result = (
        APIValidation
        .validate_openapi(
            app
        )
    )

    assert (
        result[
            "valid"
        ]
        is True
    )

    assert (
        result[
            "title"
        ]
        ==
        "Test API"
    )

    assert (
        result[
            "path_count"
        ]
        >=
        2
    )


# ============================================================
# SAFE SERVICE STATUS
# ============================================================

def test_safe_service_status():

    class DummyService:

        def status(
            self
        ):

            return {
                "available":
                    True
            }

    result = (
        APIValidation
        .safe_service_status(
            DummyService(),
            "status"
        )
    )

    assert (
        result[
            "success"
        ]
        is True
    )

    assert (
        result[
            "result"
        ][
            "available"
        ]
        is True
    )


# ============================================================
# OVERALL PASS
# ============================================================

def test_overall_status_pass():

    result = (
        APIValidation
        .overall_status(
            [
                True,
                True,
                True
            ]
        )
    )

    assert (
        result
        ==
        "PASS"
    )


# ============================================================
# OVERALL REVIEW
# ============================================================

def test_overall_status_review():

    result = (
        APIValidation
        .overall_status(
            [
                True,
                False,
                True
            ]
        )
    )

    assert (
        result
        ==
        "REVIEW_REQUIRED"
    )