import sqlite3

conn = sqlite3.connect('test.db')
cursor = conn.cursor()

# List all tables in database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables:", tables)

# Query from the first table
if tables:
    table_name = tables[0][0]
    cursor.execute(f"SELECT * FROM {table_name};")
    rows = cursor.fetchall()
    for row in rows:
        print(row)

conn.close()