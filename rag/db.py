import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_db_connection():
    try:
        connection = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
        )
        return connection
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None


def close_db_connection(connection):
    if connection:
        try:
            connection.close()
        except psycopg2.Error as e:
            print(f"Error closing the connection: {e}")
