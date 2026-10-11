from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# ============================================================
# PATHS
# ============================================================

CI_WORKFLOW = (
    PROJECT_ROOT
    / ".github"
    / "workflows"
    / "ci.yml"
)

DEPENDABOT_FILE = (
    PROJECT_ROOT
    / ".github"
    / "dependabot.yml"
)

DOCKERFILE = (
    PROJECT_ROOT
    / "Dockerfile"
)

CI_VALIDATOR = (
    PROJECT_ROOT
    / "ci_validate.py"
)


# ============================================================
# FILE EXISTENCE
# ============================================================

def test_ci_workflow_exists():

    assert CI_WORKFLOW.exists()

    assert CI_WORKFLOW.is_file()


def test_dependabot_exists():

    assert DEPENDABOT_FILE.exists()

    assert DEPENDABOT_FILE.is_file()


def test_dockerfile_exists():

    assert DOCKERFILE.exists()

    assert DOCKERFILE.is_file()


def test_ci_validator_exists():

    assert CI_VALIDATOR.exists()

    assert CI_VALIDATOR.is_file()


# ============================================================
# CI TRIGGERS
# ============================================================

def test_ci_contains_push_trigger():

    content = CI_WORKFLOW.read_text(
        encoding="utf-8"
    )

    assert "push:" in content


def test_ci_contains_pull_request_trigger():

    content = CI_WORKFLOW.read_text(
        encoding="utf-8"
    )

    assert "pull_request:" in content


def test_ci_contains_manual_trigger():

    content = CI_WORKFLOW.read_text(
        encoding="utf-8"
    )

    assert "workflow_dispatch:" in content


# ============================================================
# PYTHON TESTING
# ============================================================

def test_ci_runs_pytest():

    content = CI_WORKFLOW.read_text(
        encoding="utf-8"
    )

    assert (
        "python -m pytest"
        in content
    )


# ============================================================
# PROJECT VALIDATION
# ============================================================

def test_ci_runs_project_validation():

    content = CI_WORKFLOW.read_text(
        encoding="utf-8"
    )

    assert (
        "python ci_validate.py"
        in content
    )


# ============================================================
# DEPENDENCY VALIDATION
# ============================================================

def test_ci_runs_pip_check():

    content = CI_WORKFLOW.read_text(
        encoding="utf-8"
    )

    assert (
        "python -m pip check"
        in content
    )


# ============================================================
# SECURITY AUDIT
# ============================================================

def test_ci_contains_security_audit():

    content = CI_WORKFLOW.read_text(
        encoding="utf-8"
    )

    assert "pip-audit" in content

    assert "pip_audit" in content


# ============================================================
# DOCKER BUILD
# ============================================================

def test_ci_contains_docker_build():

    content = CI_WORKFLOW.read_text(
        encoding="utf-8"
    )

    assert (
        "docker/build-push-action"
        in content
    )


# ============================================================
# DOCKER BUILDX
# ============================================================

def test_ci_contains_buildx():

    content = CI_WORKFLOW.read_text(
        encoding="utf-8"
    )

    assert (
        "docker/setup-buildx-action"
        in content
    )


# ============================================================
# DOCKER CACHE
# ============================================================

def test_ci_contains_docker_cache():

    content = CI_WORKFLOW.read_text(
        encoding="utf-8"
    )

    assert (
        "cache-from: type=gha"
        in content
    )

    assert (
        "cache-to: type=gha"
        in content
    )


# ============================================================
# HEALTH CHECK
# ============================================================

def test_ci_validates_health():

    content = CI_WORKFLOW.read_text(
        encoding="utf-8"
    )

    assert (
        "/health"
        in content
    )


# ============================================================
# OPENAPI CHECK
# ============================================================

def test_ci_validates_openapi():

    content = CI_WORKFLOW.read_text(
        encoding="utf-8"
    )

    assert (
        "/openapi.json"
        in content
    )


# ============================================================
# ARTIFACT UPLOAD
# ============================================================

def test_ci_uploads_reports():

    content = CI_WORKFLOW.read_text(
        encoding="utf-8"
    )

    assert (
        "actions/upload-artifact"
        in content
    )


# ============================================================
# DEPENDABOT CONFIGURATION
# ============================================================

def test_dependabot_python_configuration():

    content = DEPENDABOT_FILE.read_text(
        encoding="utf-8"
    )

    assert (
        'package-ecosystem: "pip"'
        in content
    )


def test_dependabot_actions_configuration():

    content = DEPENDABOT_FILE.read_text(
        encoding="utf-8"
    )

    assert (
        'package-ecosystem: "github-actions"'
        in content
    )


def test_dependabot_docker_configuration():

    content = DEPENDABOT_FILE.read_text(
        encoding="utf-8"
    )

    assert (
        'package-ecosystem: "docker"'
        
        in content
    )