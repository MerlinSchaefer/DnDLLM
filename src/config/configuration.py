from dotenv import find_dotenv, load_dotenv
from pydantic import BaseSettings


class DatabaseConfig(BaseSettings):
    db_name: str
    host: str
    password: str
    port: str
    user: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Fetch database configuration from environment variables
db_config = DatabaseConfig()
