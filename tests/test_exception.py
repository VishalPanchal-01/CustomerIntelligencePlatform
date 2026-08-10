import sys

from src.utils.exception import CustomException


def test_custom_exception():

    try:
        10 / 0

    except Exception as e:

        custom_exception = CustomException(
            e,
            sys
        )

        print(custom_exception)

        assert "division by zero" in str(custom_exception)