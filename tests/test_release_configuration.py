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
# RELEASE WORKFLOW
# ============================================================

RELEASE_WORKFLOW = (
    PROJECT_ROOT
    / ".github"
    / "workflows"
    / "release.yml"
)


# ============================================================
# FILE EXISTS
# ============================================================

def test_release_workflow_exists():

    assert (
        RELEASE_WORKFLOW.exists()
    )

    assert (
        RELEASE_WORKFLOW.is_file()
    )


# ============================================================
# READ FILE
# ============================================================

def get_content() -> str:

    return (
        RELEASE_WORKFLOW.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# VERSION TAG TRIGGER
# ============================================================

def test_release_uses_version_tag():

    content = (
        get_content()
    )

    assert (
        '"v*"'
        in content
    )


# ============================================================
# SEMANTIC VERSION VALIDATION
# ============================================================

def test_release_checks_semantic_version():

    content = (
        get_content()
    )

    assert (
        "^[0-9]+"
        in content
        or
        "^v[0-9]+"
        in content
    )


# ============================================================
# INVALID TAG FAILURE
# ============================================================

def test_release_can_fail_invalid_tag():

    content = (
        get_content()
    )

    assert (
        "exit 1"
        in content
    )