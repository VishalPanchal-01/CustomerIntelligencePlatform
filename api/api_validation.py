import os
from typing import Any

from fastapi import FastAPI


class APIValidation:

    # =========================================================
    # FILE CHECK
    # =========================================================

    @staticmethod
    def check_file(
        path: str
    ) -> dict:

        exists = os.path.exists(
            path
        )

        return {
            "path":
                path,

            "exists":
                exists,

            "size_bytes":
                (
                    os.path.getsize(
                        path
                    )
                    if exists
                    else 0
                )
        }


    # =========================================================
    # DIRECTORY CHECK
    # =========================================================

    @staticmethod
    def check_directory(
        path: str
    ) -> dict:

        exists = os.path.isdir(
            path
        )

        return {
            "path":
                path,

            "exists":
                exists
        }


    # =========================================================
    # ROUTE LIST
    # =========================================================

    @staticmethod
    def registered_routes(
        app: FastAPI
    ) -> list[dict[str, Any]]:

        routes = []

        for route in app.routes:

            path = getattr(
                route,
                "path",
                None
            )

            methods = getattr(
                route,
                "methods",
                None
            )

            if path is None:

                continue

            routes.append(
                {
                    "path":
                        path,

                    "methods":
                        sorted(
                            list(
                                methods
                            )
                        )
                        if methods
                        else []
                }
            )

        return routes


    # =========================================================
    # ROUTE EXISTS
    # =========================================================

    @staticmethod
    def route_exists(
        app: FastAPI,
        path: str,
        method: str
    ) -> bool:

        method = method.upper()

        for route in app.routes:

            route_path = getattr(
                route,
                "path",
                None
            )

            methods = getattr(
                route,
                "methods",
                set()
            )

            if (
                route_path == path
                and
                method in methods
            ):

                return True

        return False


    # =========================================================
    # VALIDATE REQUIRED ROUTES
    # =========================================================

    def validate_required_routes(
        self,
        app: FastAPI,
        required_routes: list[tuple[str, str]]
    ) -> dict:

        details = {}

        for method, path in required_routes:

            key = (
                f"{method.upper()} {path}"
            )

            details[
                key
            ] = self.route_exists(
                app=
                    app,

                path=
                    path,

                method=
                    method
            )

        all_present = all(
            details.values()
        )

        return {
            "all_present":
                all_present,

            "routes":
                details
        }


    # =========================================================
    # OPENAPI CHECK
    # =========================================================

    @staticmethod
    def validate_openapi(
        app: FastAPI
    ) -> dict:

        try:

            schema = app.openapi()

            title = (
                schema
                .get(
                    "info",
                    {}
                )
                .get(
                    "title"
                )
            )

            version = (
                schema
                .get(
                    "info",
                    {}
                )
                .get(
                    "version"
                )
            )

            paths = (
                schema.get(
                    "paths",
                    {}
                )
            )

            return {
                "valid":
                    True,

                "title":
                    title,

                "version":
                    version,

                "path_count":
                    len(
                        paths
                    )
            }

        except Exception as error:

            return {
                "valid":
                    False,

                "error":
                    str(
                        error
                    ),

                "path_count":
                    0
            }


    # =========================================================
    # SAFE SERVICE STATUS
    # =========================================================

    @staticmethod
    def safe_service_status(
        service,
        method_name: str
    ) -> dict:

        try:

            method = getattr(
                service,
                method_name
            )

            result = method()

            return {
                "success":
                    True,

                "result":
                    result
            }

        except Exception as error:

            return {
                "success":
                    False,

                "error":
                    str(
                        error
                    )
            }


    # =========================================================
    # HEALTH SUMMARY
    # =========================================================

    @staticmethod
    def overall_status(
        conditions: list[bool]
    ) -> str:

        return (
            "PASS"
            if all(
                conditions
            )
            else
            "REVIEW_REQUIRED"
        )