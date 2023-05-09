import sqlite3

database = 'database.db'

def create_table():
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, password TEXT)')
    conn.commit()
    conn.close()

def insert_user(username, password):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
    conn.commit()
    conn.close()

def get_user(username):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    return user

def get_user_by_id(id):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute('SELECT username FROM users WHERE id = ?', (id,))
    user = cursor.fetchone()
    conn.close()
    return user

def get_all_users():
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute('SELECT username FROM users')
    users = cursor.fetchall()
    conn.close()
    return users

def update_user(id, username, password):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET username = ?, password = ? WHERE id = ?', (username, password, id))
    conn.commit()
    conn.close()

def delete_user(id):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE id = ?', (id,))
    conn.commit()
    conn.close()

def delete_all_users():
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users')
    conn.commit()
    conn.close()

def delete_user_by_username(username):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE username = ?', (username,))
    conn.commit()
    conn.close()

def main():
    insert_user('admin', 'admin')
    print(get_all_users())

if __name__ == '__main__':
    main()
    