from src.utils.logger import logger


def test_logging_check():

    logger.info("THIS IS A LOGGER TEST")

    assert True