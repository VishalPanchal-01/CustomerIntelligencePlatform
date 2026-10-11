from __future__ import annotations

import json
import subprocess
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
# OUTPUT
# ============================================================

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "final_validation"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "final_project_validation_report.json"
)


# ============================================================
# REQUIRED DIRECTORIES
# ============================================================

REQUIRED_DIRECTORIES = [
    "api",
    "dashboard",
    "src",
    "models",
    "artifacts",
    "tests",
]


# ============================================================
# REQUIRED ROOT FILES
# ============================================================

REQUIRED_ROOT_FILES = [
    "README.md",
    "requirements.txt",
    "Dockerfile",
    "docker-compose.yml",
    ".dockerignore",
    ".gitignore",
    "ci_validate.py",
]


# ============================================================
# REQUIRED API FILES
# ============================================================

REQUIRED_API_FILES = [
    "api/main.py",
    "api/schemas.py",
    "api/customer_service.py",
    "api/churn_service.py",
    "api/clv_service.py",
    "api/recommendation_service.py",
    "api/explainability_service.py",
    "api/intelligence_service.py",
]


# ============================================================
# REQUIRED MODEL ARTIFACTS
# ============================================================

REQUIRED_MODELS = [
    "models/churn/churn_model.pkl",
    "models/clv/clv_model.pkl",
    "models/clv/clv_value_bands.json",
    "models/recommendation/recommender.pkl",
    "models/recommendation/recommender_metadata.json",
]


# ============================================================
# REQUIRED BUSINESS ARTIFACTS
# ============================================================

REQUIRED_ARTIFACTS = [
    (
        "artifacts/dashboard/"
        "unified_customer_intelligence.csv"
    ),
    (
        "artifacts/recommendation/"
        "predictions/"
        "batch_recommendations.csv"
    ),
]


# ============================================================
# REQUIRED MLFLOW FILES
# ============================================================

REQUIRED_MLFLOW_FILES = [
    "src/mlflow_tracking/__init__.py",
    "src/mlflow_tracking/config.py",
    "src/mlflow_tracking/tracker.py",
    "src/mlflow_tracking/churn_tracking.py",
    "src/mlflow_tracking/clv_tracking.py",
    "src/mlflow_tracking/recommendation_tracking.py",
    "src/mlflow_tracking/mlflow_validation.py",
]


# ============================================================
# REQUIRED GITHUB WORKFLOWS
# ============================================================

REQUIRED_GITHUB_FILES = [
    ".github/workflows/ci.yml",
    ".github/workflows/cd.yml",
    ".github/workflows/release.yml",
    ".github/dependabot.yml",
]


# ============================================================
# HELPER — FILE EXISTS
# ============================================================

def path_exists(
    relative_path: str,
) -> bool:

    return (
        PROJECT_ROOT
        / relative_path
    ).exists()


# ============================================================
# VALIDATE PATH GROUP
# ============================================================

def validate_paths(
    paths: list[str],
) -> dict[str, Any]:

    results = {}

    for relative_path in paths:

        results[
            relative_path
        ] = path_exists(
            relative_path
        )

    return {
        "success":
            all(
                results.values()
            ),

        "items":
            results,
    }


# ============================================================
# VALIDATE DIRECTORIES
# ============================================================

def validate_directories(
) -> dict[str, Any]:

    results = {}

    for directory in (
        REQUIRED_DIRECTORIES
    ):

        path = (
            PROJECT_ROOT
            / directory
        )

        results[
            directory
        ] = (
            path.exists()
            and
            path.is_dir()
        )

    return {
        "success":
            all(
                results.values()
            ),

        "items":
            results,
    }


# ============================================================
# RUN COMMAND
# ============================================================

def run_command(
    command: list[str],
) -> dict[str, Any]:

    try:

        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        return {
            "success":
                result.returncode == 0,

            "return_code":
                result.returncode,

            "stdout":
                result.stdout.strip(),

            "stderr":
                result.stderr.strip(),
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
                str(error),
        }


# ============================================================
# PYTHON VERSION
# ============================================================

def validate_python(
) -> dict[str, Any]:

    result = run_command(
        [
            sys.executable,
            "--version",
        ]
    )

    return {
        "success":
            result[
                "success"
            ],

        "version":
            (
                result[
                    "stdout"
                ]
                or
                result[
                    "stderr"
                ]
            ),
    }


# ============================================================
# PYTEST
# ============================================================

def validate_pytest(
) -> dict[str, Any]:

    result = run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
        ]
    )

    return result


# ============================================================
# PROJECT CI VALIDATOR
# ============================================================

def validate_ci_script(
) -> dict[str, Any]:

    if not (
        PROJECT_ROOT
        / "ci_validate.py"
    ).exists():

        return {
            "success":
                False,

            "return_code":
                -1,

            "stdout":
                "",

            "stderr":
                "ci_validate.py not found.",
        }

    return run_command(
        [
            sys.executable,
            "ci_validate.py",
        ]
    )


# ============================================================
# REQUIREMENTS
# ============================================================

def validate_requirements(
) -> dict[str, Any]:

    requirements_path = (
        PROJECT_ROOT
        / "requirements.txt"
    )

    if not requirements_path.exists():

        return {
            "success":
                False,

            "detail":
                "requirements.txt missing.",
        }

    content = (
        requirements_path
        .read_text(
            encoding="utf-8"
        )
        .lower()
    )

    expected_packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "fastapi",
        "uvicorn",
        "pydantic",
        "joblib",
        "mlflow",
        "shap",
    ]

    package_results = {}

    for package in (
        expected_packages
    ):

        package_results[
            package
        ] = (
            package
            in content
        )

    return {
        "success":
            all(
                package_results.values()
            ),

        "packages":
            package_results,
    }


# ============================================================
# GITIGNORE
# ============================================================

def validate_gitignore(
) -> dict[str, Any]:

    path = (
        PROJECT_ROOT
        / ".gitignore"
    )

    if not path.exists():

        return {
            "success":
                False,

            "checks":
                {},
        }

    content = (
        path.read_text(
            encoding="utf-8"
        )
        .lower()
    )

    checks = {
        ".venv":
            (
                ".venv"
                in content
            ),

        "__pycache__":
            (
                "__pycache__"
                in content
            ),

        ".env":
            (
                ".env"
                in content
            ),

        "mlruns":
            (
                "mlruns"
                in content
            ),
    }

    return {
        "success":
            all(
                checks.values()
            ),

        "checks":
            checks,
    }


# ============================================================
# DOCKERIGNORE
# ============================================================

def validate_dockerignore(
) -> dict[str, Any]:

    path = (
        PROJECT_ROOT
        / ".dockerignore"
    )

    if not path.exists():

        return {
            "success":
                False,

            "checks":
                {},
        }

    content = (
        path.read_text(
            encoding="utf-8"
        )
        .lower()
    )

    checks = {
        ".venv":
            (
                ".venv"
                in content
            ),

        ".git":
            (
                ".git"
                in content
            ),

        "notebooks":
            (
                "notebooks/"
                in content
            ),

        "mlruns":
            (
                "mlruns/"
                in content
            ),

        "data":
            (
                "data/"
                in content
            ),
    }

    return {
        "success":
            all(
                checks.values()
            ),

        "checks":
            checks,
    }


# ============================================================
# README
# ============================================================

def validate_readme(
) -> dict[str, Any]:

    path = (
        PROJECT_ROOT
        / "README.md"
    )

    if not path.exists():

        return {
            "success":
                False,

            "checks":
                {},
        }

    content = (
        path.read_text(
            encoding="utf-8"
        )
        .lower()
    )

    required_topics = {
        "churn":
            "churn",

        "segmentation":
            "segmentation",

        "customer value":
            "90-day",

        "recommendation":
            "recommendation",

        "shap":
            "shap",

        "fastapi":
            "fastapi",

        "docker":
            "docker",

        "mlflow":
            "mlflow",

        "ci/cd":
            "ci/cd",
    }

    checks = {}

    for name, keyword in (
        required_topics.items()
    ):

        checks[
            name
        ] = (
            keyword
            in content
        )

    return {
        "success":
            all(
                checks.values()
            ),

        "checks":
            checks,
    }


# ============================================================
# GIT STATUS
# ============================================================

def validate_git_status(
) -> dict[str, Any]:

    result = run_command(
        [
            "git",
            "status",
            "--porcelain",
        ]
    )

    if not result[
        "success"
    ]:

        return result

    clean = (
        result[
            "stdout"
        ].strip()
        ==
        ""
    )

    return {
        "success":
            clean,

        "clean":
            clean,

        "changes":
            result[
                "stdout"
            ],
    }


# ============================================================
# GIT LFS
# ============================================================

def validate_git_lfs(
) -> dict[str, Any]:

    result = run_command(
        [
            "git",
            "lfs",
            "ls-files",
        ]
    )

    return {
        "success":
            result[
                "success"
            ],

        "files":
            result[
                "stdout"
            ],

        "error":
            result[
                "stderr"
            ],
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
        "FINAL PROJECT VALIDATION"
    )

    print(
        "========================================"
        "\n"
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    validations = {
        "directories":
            validate_directories(),

        "root_files":
            validate_paths(
                REQUIRED_ROOT_FILES
            ),

        "api_files":
            validate_paths(
                REQUIRED_API_FILES
            ),

        "model_artifacts":
            validate_paths(
                REQUIRED_MODELS
            ),

        "business_artifacts":
            validate_paths(
                REQUIRED_ARTIFACTS
            ),

        "mlflow_files":
            validate_paths(
                REQUIRED_MLFLOW_FILES
            ),

        "github_files":
            validate_paths(
                REQUIRED_GITHUB_FILES
            ),

        "python":
            validate_python(),

        "requirements":
            validate_requirements(),

        "gitignore":
            validate_gitignore(),

        "dockerignore":
            validate_dockerignore(),

        "readme":
            validate_readme(),

        "pytest":
            validate_pytest(),

        "ci_validation":
            validate_ci_script(),

        "git_status":
            validate_git_status(),

        "git_lfs":
            validate_git_lfs(),
    }

    # --------------------------------------------------------
    # Git status is useful information but a dirty working
    # directory should not make technical validation fail.
    # --------------------------------------------------------

    blocking_sections = [
        "directories",
        "root_files",
        "api_files",
        "model_artifacts",
        "business_artifacts",
        "mlflow_files",
        "github_files",
        "python",
        "requirements",
        "gitignore",
        "dockerignore",
        "readme",
        "pytest",
        "ci_validation",
    ]

    overall_success = all(
        validations[
            section
        ][
            "success"
        ]
        for section
        in blocking_sections
    )

    report = {
        "project":
            (
                "AI-Powered Customer "
                "Intelligence Platform"
            ),

        "overall_status":
            (
                "PASS"
                if overall_success
                else
                "REVIEW_REQUIRED"
            ),

        "validations":
            validations,
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    for name, result in (
        validations.items()
    ):

        status = (
            "PASS"
            if result.get(
                "success",
                False,
            )
            else
            "REVIEW"
        )

        print(
            f"{name:<25} {status}"
        )

    print(
        "\n"
        "========================================"
    )

    print(
        "FINAL STATUS"
    )

    print(
        "========================================"
    )

    print(
        report[
            "overall_status"
        ]
    )

    print(
        "\nReport saved to:"
    )

    print(
        REPORT_PATH
    )

    # ========================================================
    # GIT STATUS MESSAGE
    # ========================================================

    git_status = validations[
        "git_status"
    ]

    if git_status.get(
        "success"
    ):

        print(
            "\nGit working tree is clean."
        )

    else:

        print(
            "\nGit working tree "
            "contains changes."
        )

        changes = git_status.get(
            "changes"
        )

        if changes:

            print(
                changes
            )

    # ========================================================
    # EXIT CODE
    # ========================================================

    if not overall_success:

        print(
            "\nProject requires review "
            "before final completion."
        )

        sys.exit(
            1
        )

    print(
        "\nFINAL PROJECT VALIDATION PASSED."
    )

    sys.exit(
        0
    )


if __name__ == "__main__":

    main()