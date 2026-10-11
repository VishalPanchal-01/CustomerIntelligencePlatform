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
# FILE PATH
# ============================================================

CD_WORKFLOW = (
    PROJECT_ROOT
    / ".github"
    / "workflows"
    / "cd.yml"
)


# ============================================================
# FILE EXISTENCE
# ============================================================

def test_cd_workflow_exists():

    assert (
        CD_WORKFLOW.exists()
    )

    assert (
        CD_WORKFLOW.is_file()
    )


# ============================================================
# LOAD WORKFLOW
# ============================================================

def get_workflow_content() -> str:

    return (
        CD_WORKFLOW.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# MAIN BRANCH
# ============================================================

def test_cd_targets_main():

    content = (
        get_workflow_content()
    )

    assert "main" in content


# ============================================================
# VERSION TAGS
# ============================================================

def test_cd_supports_version_tags():

    content = (
        get_workflow_content()
    )

    assert (
        '"v*"'
        in content
    )


# ============================================================
# MANUAL TRIGGER
# ============================================================

def test_cd_supports_manual_execution():

    content = (
        get_workflow_content()
    )

    assert (
        "workflow_dispatch:"
        in content
    )


# ============================================================
# PACKAGE WRITE PERMISSION
# ============================================================

def test_cd_has_package_write_permission():

    content = (
        get_workflow_content()
    )

    assert (
        "packages: write"
        in content
    )


# ============================================================
# GHCR
# ============================================================

def test_cd_uses_ghcr():

    content = (
        get_workflow_content()
    )

    assert (
        "ghcr.io"
        in content
    )


# ============================================================
# GITHUB TOKEN
# ============================================================

def test_cd_uses_github_token():

    content = (
        get_workflow_content()
    )

    assert (
        "secrets.GITHUB_TOKEN"
        in content
    )


# ============================================================
# DOCKER LOGIN
# ============================================================

def test_cd_uses_docker_login():

    content = (
        get_workflow_content()
    )

    assert (
        "docker/login-action"
        in content
    )


# ============================================================
# DOCKER METADATA
# ============================================================

def test_cd_uses_docker_metadata():

    content = (
        get_workflow_content()
    )

    assert (
        "docker/metadata-action"
        in content
    )


# ============================================================
# BUILDX
# ============================================================

def test_cd_uses_buildx():

    content = (
        get_workflow_content()
    )

    assert (
        "docker/setup-buildx-action"
        in content
    )


# ============================================================
# BUILD PUSH
# ============================================================

def test_cd_builds_and_pushes_image():

    content = (
        get_workflow_content()
    )

    assert (
        "docker/build-push-action"
        in content
    )

    assert (
        "push: true"
        in content
    )


# ============================================================
# LATEST TAG
# ============================================================

def test_cd_contains_latest_tag():

    content = (
        get_workflow_content()
    )

    assert (
        "type=raw,value=latest"
        in content
    )


# ============================================================
# COMMIT SHA TAG
# ============================================================

def test_cd_contains_sha_tag():

    content = (
        get_workflow_content()
    )

    assert (
        "type=sha"
        in content
    )


# ============================================================
# RELEASE TAG
# ============================================================

def test_cd_contains_release_tag():

    content = (
        get_workflow_content()
    )

    assert (
        "type=ref,event=tag"
        in content
    )


# ============================================================
# DOCKER CACHE
# ============================================================

def test_cd_uses_docker_cache():

    content = (
        get_workflow_content()
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
# POST-PUBLISH VERIFICATION
# ============================================================

def test_cd_verifies_published_image():

    content = (
        get_workflow_content()
    )

    assert (
        "docker pull"
        in content
    )

    assert (
        "/health"
        in content
    )


# ============================================================
# OPENAPI VALIDATION
# ============================================================

def test_cd_validates_openapi():

    content = (
        get_workflow_content()
    )

    assert (
        "/openapi.json"
        in content
    )


# ============================================================
# MODEL VALIDATION
# ============================================================

def test_cd_validates_models():

    content = (
        get_workflow_content()
    )

    assert (
        "/models/churn/status"
        in content
    )

    assert (
        "/models/clv/status"
        in content
    )

    assert (
        "/models/recommendation/status"
        in content
    )


# ============================================================
# VALIDATION BEFORE PUBLISHING
# ============================================================

def test_cd_runs_tests_before_publish():

    content = (
        get_workflow_content()
    )

    assert (
        "python -m pytest"
        in content
    )

    assert (
        "python ci_validate.py"
        in content
    )