import sqlite3

def init_all_tables():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # 1. 建立會員表 
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    
    # 2. 建立紀錄表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            correct_count INTEGER,
            error_count INTEGER,
            duration_seconds INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("會員表與運動紀錄表皆已同步建立成功！")

if __name__ == "__main__":
    init_all_tables()