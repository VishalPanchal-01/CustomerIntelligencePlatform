import os

from dotenv import load_dotenv


load_dotenv()


PROJECT_ENV = os.getenv(
    "PROJECT_ENV",
    "development"
)