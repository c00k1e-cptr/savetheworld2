import sqlite3

database = 'database.db'


def create_table_users():
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, password TEXT, first_name TEXT, last_name TEXT, age INTEGER, height INTEGER, weight INTEGER, gender FLOAT);")
    conn.commit()
    conn.close()


def create_table_results():
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS results (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, test_name TEXT, result FLOAT, date TEXT)""")
    conn.commit()
    conn.close()


def insert_user(username, password, first_name, last_name, age, height, weight, gender):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO users (username, password, first_name, last_name, age, height, weight, gender) VALUES (?, ?, ?, ?, ?, ?, ?, ?)', (username, password, first_name, last_name, age, height, weight, gender))
    conn.commit()
    conn.close()


def get_user(username):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    return user


def get_all_users():
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute('SELECT username, password FROM users')
    users = cursor.fetchall()
    conn.close()
    tmp = {}
    for user in users:
        tmp[user[0]] = user[1]
    return tmp


def update_user(id, username, password):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE users SET username = ?, password = ? WHERE username = ?', (username, password, username))
    conn.commit()
    conn.close()


def delete_user_by_username(username):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE username = ?', (username,))
    conn.commit()
    conn.close()


def get_all_user_data():
    # debugging purposes
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users')
    users = cursor.fetchall()
    conn.close()
    return users


def preload():
    create_table_users()
    create_table_results()
    print(get_all_user_data())


# preload()
