# import logging
# import os
# from datetime import datetime


# LOG_DIR = "logs"

# os.makedirs(
#     LOG_DIR,
#     exist_ok=True
# )

# LOG_FILE = os.path.join(
#     LOG_DIR,
#     f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# )


# logging.basicConfig(
#     filename=LOG_FILE,
#     format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
#     level=logging.INFO,
# )


# logger = logging.getLogger(__name__)



import logging
import os
from datetime import datetime


LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(
    LOG_DIR,
    f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
)


logger = logging.getLogger("customer_intelligence")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILE)

formatter = logging.Formatter(
    "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)

file_handler.setFormatter(formatter)

logger.addHandler(file_handler)