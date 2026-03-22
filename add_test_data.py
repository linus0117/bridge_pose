import sqlite3

def add_mock_data():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    user = 'abc' 
    
    test_records = [
        (user, 10, 2, 60, '2026-03-16 10:00:00'),
        (user, 25, 5, 120, '2026-03-17 11:00:00'),
        (user, 18, 3, 90, '2026-03-18 09:30:00'),
        (user, 30, 1, 150, '2026-03-19 20:00:00'),
        (user, 22, 4, 100, '2026-03-20 12:00:00')
    ]
    
    cursor.executemany("""
        INSERT INTO records (username, correct_count, error_count, duration_seconds, timestamp) 
        VALUES (?, ?, ?, ?, ?)
    """, test_records)
    
    conn.commit()
    conn.close()
    print(f"資料更新完成")

if __name__ == "__main__":
    add_mock_data()