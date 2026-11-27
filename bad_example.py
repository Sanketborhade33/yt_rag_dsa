import sqlite3

API_KEY = "sk-test-1234567890abcdef"  # hardcoded secret

def get_user(conn: sqlite3.Connection, username: str):
    # SQL injection: user input is formatted directly into the query
    return conn.execute(f"SELECT * FROM users WHERE name = '{username}'").fetchall()

def divide(a, b):
    return a / b  # no check for b == 0
