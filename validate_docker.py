import json
import os
import subprocess
from pathlib import Path
from typing import Any


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

REPORT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "docker"
)

REPORT_PATH = (
    REPORT_DIR
    / "docker_report.json"
)

CONTAINER_NAME = (
    "customer-intelligence-api-container"
)

IMAGE_NAME = (
    "customer-intelligence-api:latest"
)

API_URL = (
    "http://127.0.0.1:8000"
)


# ============================================================
# COMMAND RUNNER
# ============================================================

def run_command(
    command: list[str]
) -> dict[str, Any]:

    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False
        )

        return {
            "success":
                result.returncode == 0,

            "return_code":
                result.returncode,

            "stdout":
                result.stdout.strip(),

            "stderr":
                result.stderr.strip()
        }

    except Exception as error:

        return {
            "success":
                False,

            "return_code":
                -1,

            "stdout":
                "",

            "stderr":
                str(error)
        }


# ============================================================
# CONTAINER EXISTS
# ============================================================

def container_exists() -> bool:

    result = run_command(
        [
            "docker",
            "inspect",
            CONTAINER_NAME
        ]
    )

    return result["success"]


# ============================================================
# CONTAINER RUNNING
# ============================================================

def container_running() -> bool:

    result = run_command(
        [
            "docker",
            "inspect",
            "--format={{.State.Running}}",
            CONTAINER_NAME
        ]
    )

    return (
        result["success"]
        and
        result["stdout"].lower()
        ==
        "true"
    )


# ============================================================
# CONTAINER HEALTH
# ============================================================

def container_health() -> str:

    result = run_command(
        [
            "docker",
            "inspect",
            "--format={{.State.Health.Status}}",
            CONTAINER_NAME
        ]
    )

    if not result["success"]:

        return "unknown"

    return (
        result["stdout"]
        or
        "unknown"
    )


# ============================================================
# IMAGE EXISTS
# ============================================================

def image_exists() -> bool:

    result = run_command(
        [
            "docker",
            "image",
            "inspect",
            IMAGE_NAME
        ]
    )

    return result["success"]


# ============================================================
# IMAGE SIZE
# ============================================================

def image_size_bytes() -> int:

    result = run_command(
        [
            "docker",
            "image",
            "inspect",
            "--format={{.Size}}",
            IMAGE_NAME
        ]
    )

    if not result["success"]:

        return 0

    try:

        return int(
            result["stdout"]
        )

    except ValueError:

        return 0


# ============================================================
# HUMAN SIZE
# ============================================================

def format_bytes(
    size: int
) -> str:

    value = float(
        size
    )

    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB"
    ]

    for unit in units:

        if value < 1024:

            return (
                f"{value:.2f} {unit}"
            )

        value /= 1024

    return (
        f"{value:.2f} PB"
    )


# ============================================================
# FILE CHECK INSIDE CONTAINER
# ============================================================

def container_file_exists(
    path: str
) -> bool:

    result = run_command(
        [
            "docker",
            "exec",
            CONTAINER_NAME,
            "test",
            "-f",
            path
        ]
    )

    return result["success"]


# ============================================================
# ENVIRONMENT VARIABLE
# ============================================================

def container_env(
    variable: str
) -> str | None:

    result = run_command(
        [
            "docker",
            "exec",
            CONTAINER_NAME,
            "printenv",
            variable
        ]
    )

    if not result["success"]:

        return None

    return result["stdout"]


# ============================================================
# PYTHON VERSION
# ============================================================

def python_version() -> str | None:

    result = run_command(
        [
            "docker",
            "exec",
            CONTAINER_NAME,
            "python",
            "--version"
        ]
    )

    if not result["success"]:

        return None

    return (
        result["stdout"]
        or
        result["stderr"]
    )


# ============================================================
# WORKING DIRECTORY
# ============================================================

def working_directory() -> str | None:

    result = run_command(
        [
            "docker",
            "exec",
            CONTAINER_NAME,
            "pwd"
        ]
    )

    if not result["success"]:

        return None

    return result["stdout"]


# ============================================================
# HTTP CHECK FROM HOST
# ============================================================

def http_check(
    path: str
) -> dict[str, Any]:

    url = (
        f"{API_URL}{path}"
    )

    powershell_command = (
        "$response = Invoke-WebRequest "
        f"-Uri '{url}' "
        "-UseBasicParsing "
        "-TimeoutSec 10; "
        "Write-Output $response.StatusCode"
    )

    result = run_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            powershell_command
        ]
    )

    status_code = None

    if result["success"]:

        try:

            status_code = int(
                result["stdout"]
                .splitlines()[-1]
                .strip()
            )

        except (
            ValueError,
            IndexError
        ):

            status_code = None

    return {
        "url":
            url,

        "success":
            (
                result["success"]
                and
                status_code == 200
            ),

        "status_code":
            status_code,

        "error":
            (
                result["stderr"]
                if not result["success"]
                else None
            )
    }


# ============================================================
# IMPORTANT ARTIFACTS
# ============================================================

IMPORTANT_FILES = {

    "churn_model":
        "/app/models/churn/churn_model.pkl",

    "clv_model":
        "/app/models/clv/clv_model.pkl",

    "clv_value_bands":
        "/app/models/clv/clv_value_bands.json",

    "recommendation_model":
        "/app/models/recommendation/recommender.pkl",

    "customer_intelligence":
        (
            "/app/artifacts/dashboard/"
            "unified_customer_intelligence.csv"
        ),

    "recommendation_batch":
        (
            "/app/artifacts/recommendation/"
            "predictions/batch_recommendations.csv"
        ),

    "churn_global_shap":
        (
            "/app/artifacts/explainability/churn/"
            "churn_global_shap_importance.csv"
        ),

    "churn_local_shap":
        (
            "/app/artifacts/explainability/churn/"
            "churn_dashboard_shap_values.csv"
        ),

    "clv_global_shap":
        (
            "/app/artifacts/explainability/clv/"
            "clv_global_shap_importance.csv"
        ),

    "clv_local_shap":
        (
            "/app/artifacts/explainability/clv/"
            "clv_dashboard_shap_values.csv"
        )
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
        "DOCKER VALIDATION"
    )

    print(
        "========================================"
        "\n"
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # IMAGE
    # ========================================================

    image_available = (
        image_exists()
    )

    image_size = (
        image_size_bytes()
        if image_available
        else 0
    )

    # ========================================================
    # CONTAINER
    # ========================================================

    exists = (
        container_exists()
    )

    running = (
        container_running()
        if exists
        else False
    )

    health = (
        container_health()
        if exists
        else "missing"
    )

    # ========================================================
    # ENVIRONMENT
    # ========================================================

    environment = {

        "python_version":
            (
                python_version()
                if running
                else None
            ),

        "working_directory":
            (
                working_directory()
                if running
                else None
            ),

        "app_env":
            (
                container_env(
                    "APP_ENV"
                )
                if running
                else None
            ),

        "python_unbuffered":
            (
                container_env(
                    "PYTHONUNBUFFERED"
                )
                if running
                else None
            )
    }

    # ========================================================
    # ARTIFACTS
    # ========================================================

    artifact_validation = {}

    for name, path in (
        IMPORTANT_FILES.items()
    ):

        artifact_validation[
            name
        ] = {
            "path":
                path,

            "exists":
                (
                    container_file_exists(
                        path
                    )
                    if running
                    else False
                )
        }

    # ========================================================
    # ENDPOINTS
    # ========================================================

    endpoint_paths = [
        "/health",
        "/data/status",
        "/models/churn/status",
        "/models/clv/status",
        "/models/recommendation/status",
        "/explainability/status"
    ]

    endpoint_validation = {}

    for path in endpoint_paths:

        endpoint_validation[
            path
        ] = (
            http_check(
                path
            )
            if running
            else {
                "url":
                    f"{API_URL}{path}",

                "success":
                    False,

                "status_code":
                    None,

                "error":
                    "Container is not running."
            }
        )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    all_artifacts = all(
        item[
            "exists"
        ]
        for item
        in artifact_validation.values()
    )

    all_endpoints = all(
        item[
            "success"
        ]
        for item
        in endpoint_validation.values()
    )

    docker_ready = all(
        [
            image_available,
            exists,
            running,
            health == "healthy",
            all_artifacts,
            all_endpoints
        ]
    )

    overall_status = (
        "PASS"
        if docker_ready
        else
        "REVIEW_REQUIRED"
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {

        "phase":
            "Phase 8 — Docker",

        "status":
            overall_status,

        "image": {

            "name":
                IMAGE_NAME,

            "available":
                image_available,

            "size_bytes":
                image_size,

            "size_human":
                format_bytes(
                    image_size
                )
        },

        "container": {

            "name":
                CONTAINER_NAME,

            "exists":
                exists,

            "running":
                running,

            "health":
                health,

            "host_port":
                8000,

            "container_port":
                8000
        },

        "environment":
            environment,

        "artifact_validation":
            artifact_validation,

        "endpoint_validation":
            endpoint_validation,

        "deployment_configuration": {

            "container_runtime":
                "Docker Desktop / Linux container",

            "api_server":
                "Uvicorn",

            "api_host":
                "0.0.0.0",

            "health_endpoint":
                "/health",

            "compose_supported":
                True
        },

        "known_considerations": [

            (
                "The current Docker image includes "
                "ML dependencies and model artifacts, "
                "so the image is expected to be larger "
                "than a basic FastAPI image."
            ),

            (
                "The image should be rebuilt whenever "
                "application code, model artifacts, or "
                "Python dependencies change."
            ),

            (
                "Pinned dependency versions are recommended "
                "for reproducible loading of persisted "
                "scikit-learn/XGBoost model artifacts."
            ),

            (
                "The development CORS policy should be "
                "restricted before a public production "
                "deployment."
            )
        ],

        "validation_summary": {

            "image_available":
                image_available,

            "container_exists":
                exists,

            "container_running":
                running,

            "container_healthy":
                health == "healthy",

            "required_artifacts_present":
                all_artifacts,

            "core_endpoints_respond":
                all_endpoints,

            "overall_status":
                overall_status
        }
    }

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
        "\nImage:"
    )

    print(
        f"- Available: {image_available}"
    )

    print(
        (
            "- Size: "
            f"{format_bytes(image_size)}"
        )
    )

    print(
        "\nContainer:"
    )

    print(
        f"- Exists: {exists}"
    )

    print(
        f"- Running: {running}"
    )

    print(
        f"- Health: {health}"
    )

    print(
        "\nArtifacts:"
    )

    for name, result in (
        artifact_validation.items()
    ):

        print(
            f"- {name}: "
            f"{'OK' if result['exists'] else 'MISSING'}"
        )

    print(
        "\nEndpoints:"
    )

    for path, result in (
        endpoint_validation.items()
    ):

        print(
            f"- {path}: "
            f"{result['status_code']}"
        )

    print(
        "\nReport saved:"
    )

    print(
        REPORT_PATH
    )

    if overall_status == "PASS":

        print(
            "\nDOCKER VALIDATION PASSED."
        )

    else:

        print(
            "\nDocker configuration requires review."
        )


if __name__ == "__main__":

    main()