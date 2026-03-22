import sqlite3

def init_db():
    # 建立一個叫 users.db 的資料庫檔案
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # 建立會員資料表
    # id: 自動編號
    # username: 帳號 (不能重複)
    # password: 密碼
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()
    print("會員資料庫初始化成功！")

if __name__ == "__main__":
    init_db()