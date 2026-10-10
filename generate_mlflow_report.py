from src.mlflow_tracking.mlflow_validation import (
    FINAL_REPORT_PATH,
    RUN_SUMMARY_PATH,
    MLflowValidation,
    save_report,
)


def print_experiment_result(
    result: dict,
) -> None:

    print(
        f"\nExperiment: "
        f"{result['experiment_name']}"
    )

    print(
        f"Exists: "
        f"{result['exists']}"
    )

    print(
        f"Experiment ID: "
        f"{result['experiment_id']}"
    )

    print(
        f"Run Count: "
        f"{result['run_count']}"
    )

    print(
        f"Finished Runs: "
        f"{result['finished_run_count']}"
    )

    print(
        f"Failed Runs: "
        f"{result['failed_run_count']}"
    )

    print(
        "Production Run Present: "
        f"{result['production_run_present']}"
    )

    print(
        f"Status: "
        f"{result['status']}"
    )


def main():

    print(
        "\n"
        "========================================"
    )

    print(
        "MLFLOW FINAL VALIDATION"
    )

    print(
        "========================================"
        "\n"
    )

    validator = (
        MLflowValidation()
    )

    report, summary_df = (
        validator.validate_all()
    )

    save_report(
        report=
            report,

        summary_df=
            summary_df,
    )

    for result in (
        report[
            "experiments"
        ].values()
    ):

        print_experiment_result(
            result
        )

    print(
        "\n"
        "========================================"
    )

    print(
        "SUMMARY"
    )

    print(
        "========================================"
    )

    validation = report[
        "validation_summary"
    ]

    print(
        "All experiments exist: "
        f"{validation['all_experiments_exist']}"
    )

    print(
        "All experiments have runs: "
        f"{validation['all_experiments_have_runs']}"
    )

    print(
        "All production runs present: "
        f"{validation['all_production_runs_present']}"
    )

    print(
        "Failed MLflow runs: "
        f"{validation['total_failed_runs']}"
    )

    print(
        "\nOverall Status: "
        f"{report['status']}"
    )

    print(
        "\nFinal report:"
    )

    print(
        FINAL_REPORT_PATH
    )

    print(
        "\nRun summary:"
    )

    print(
        RUN_SUMMARY_PATH
    )

    if not summary_df.empty:

        print(
            "\n"
            "========================================"
        )

        print(
            "RUNS"
        )

        print(
            "========================================"
        )

        print(
            summary_df.to_string(
                index=False
            )
        )

    if report[
        "status"
    ] == "PASS":

        print(
            "\nPHASE 9 MLFLOW VALIDATION PASSED."
        )

    else:

        print(
            "\nMLflow configuration "
            "requires review."
        )


if __name__ == "__main__":

    main()