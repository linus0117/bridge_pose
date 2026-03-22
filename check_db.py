import sqlite3

def check_data():
    # 連接到資料庫
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # 查詢所有的資料
    cursor.execute("SELECT * FROM users")
    rows = cursor.fetchall()

    print("--- 目前資料庫內的會員資料 ---")
    for row in rows:
        print(f"ID: {row[0]}, 帳號: {row[1]}, 密碼: {row[2]}")
    

    cursor.execute("SELECT * FROM records")
    records = cursor.fetchall()
    print("\n--- 目前資料庫內的運動紀錄 ---")
    for record in records:
        print(f"Timestamp: {record[5]}, 使用者: {record[1]}, 正確次數: {record[2]}, 錯誤次數: {record[3]}, 運動時間 (秒): {record[4]}")
    conn.close()

if __name__ == "__main__":
    check_data()