import os
import sqlite3

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'archive.db')

def init_db(db_path_arg=None):
    path = db_path_arg if db_path_arg is not None else globals().get('db_path')
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        video_name TEXT,
        timestamp TEXT,
        duration REAL,
        total_frames INTEGER,
        total_events INTEGER,
        confirmed_collisions INTEGER,
        visual_accidents INTEGER,
        actual_fps REAL,
        status TEXT,
        video_url TEXT,
        report_url TEXT,
        csv_url TEXT,
        summary_json TEXT,
        date TEXT,
        total_incidents INTEGER,
        duration_secs REAL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        frame INTEGER,
        ts REAL,
        level TEXT,
        type TEXT,
        ids TEXT,
        score REAL,
        message TEXT,
        bbox TEXT,
        FOREIGN KEY(run_id) REFERENCES runs(run_id)
    )
    """)
    conn.commit()
    conn.close()

def save_run(run_data, db_path=None):
    path = db_path if db_path is not None else globals().get('db_path')
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA busy_timeout=5000")
    
    run_id = run_data.get('run_id')
    cursor.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,))
    exists = cursor.fetchone()
    
    if exists:
        # Update existing row
        update_cols = [col for col in run_data.keys() if col != 'run_id']
        if update_cols:
            set_clause = ', '.join([f"{col} = ?" for col in update_cols])
            values = [run_data[col] for col in update_cols]
            values.append(run_id)
            sql = f"UPDATE runs SET {set_clause} WHERE run_id = ?"
            cursor.execute(sql, values)
    else:
        # Insert new row
        columns = ', '.join(run_data.keys())
        placeholders = ', '.join(['?'] * len(run_data))
        sql = f"INSERT INTO runs ({columns}) VALUES ({placeholders})"
        cursor.execute(sql, list(run_data.values()))
        
    conn.commit()
    conn.close()

def save_event(event_data, db_path=None):
    path = db_path if db_path is not None else globals().get('db_path')
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA busy_timeout=5000")
    columns = ', '.join(event_data.keys())
    placeholders = ', '.join(['?'] * len(event_data))
    sql = f"INSERT INTO events ({columns}) VALUES ({placeholders})"
    cursor.execute(sql, list(event_data.values()))
    conn.commit()
    conn.close()

def get_history(limit=20, db_path=None):
    path = db_path if db_path is not None else globals().get('db_path')
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    history = [dict(row) for row in rows]
    conn.close()
    return history
