import os
from datetime import datetime

import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Fetch database configuration from environment variables
db_name = os.getenv("DB_NAME")
host = os.getenv("DB_HOST")
password = os.getenv("DB_PASSWORD")
port = os.getenv("DB_PORT")
user = os.getenv("DB_USER")


# Connect to Postgres and create the database
conn = psycopg2.connect(
    dbname="postgres",
    host=host,
    password=password,
    port=port,
    user=user,
)
conn.autocommit = True

try:
    with conn.cursor() as c:
        # Check if the database exists and create it only if it doesn't
        c.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
        exists = c.fetchone()
        if not exists:
            c.execute(f"CREATE DATABASE {db_name}")
            print(f"Database '{db_name}' created.")
        else:
            print(f"Database '{db_name}' already exists.")
            c.execute("CREATE SCHEMA IF NOT EXISTS public;")
            c.execute("CREATE TABLE IF NOT EXISTS public.dev_table (id serial PRIMARY KEY, testname VARCHAR(50));")
            today_date = f"test:{datetime.now().strftime('%Y-%m-%d %H:%M')}"  # noqa
            c.execute("INSERT INTO public.dev_table (testname) VALUES (%s);", (today_date,))
            c.execute("SELECT * FROM dev_table; ")

            table = c.fetchall()
            print(table)
finally:
    conn.close()
# Close the initial connection
conn.close()
