import os

from dotenv import find_dotenv, load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    db_name: str = Field(default=os.environ.get("DB_NAME", "default_db"))
    host: str = Field(default=os.environ.get("HOST", "localhost"))
    password: str = Field(default=os.environ.get("DB_PASSWORD", "default_password"))
    port: str = Field(default=os.environ.get("PORT", "5432"))
    user: str = Field(default=os.environ.get("DB_USER", "default_user"))


# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Fetch database configuration from environment variables
db_config = DatabaseConfig()
