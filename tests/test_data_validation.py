from src.validation.data_validation import DataValidation


def test_data_validation():

    data_validation = DataValidation()

    result = data_validation.validate_data()

    assert result is True