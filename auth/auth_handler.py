import sqlite3
import hashlib
import secrets
import os
import json

# Database path is set relative to this file's directory (parent/users.db)
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "users.db")

def init_db():
    """Initializes the database and creates the users table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            chunks_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(username) REFERENCES users(username)
        );
    """)
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    """Hash a password using PBKDF2-SHA256 with a random salt."""
    salt = secrets.token_bytes(16)
    iterations = 100000
    dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iterations)
    return f"{salt.hex()}${iterations}${dk.hex()}"

def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored hash."""
    try:
        salt_hex, iterations_str, hash_hex = stored_hash.split('$')
        salt = bytes.fromhex(salt_hex)
        iterations = int(iterations_str)
        dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iterations)
        return secrets.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False

def register_user(username: str, password: str) -> tuple[bool, str]:
    """Register a new user in the database."""
    username = username.strip()
    if not username or not password:
        return False, "Username and password cannot be empty."
    
    init_db() # Ensure DB is initialized
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        # Check if user already exists
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            conn.close()
            return False, "Username already exists."
        
        # Hash password and insert
        hashed = hash_password(password)
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
        conn.close()
        return True, "User registered successfully."
    except sqlite3.Error as e:
        conn.close()
        return False, f"Database error: {str(e)}"

def authenticate_user(username: str, password: str) -> tuple[bool, str]:
    """Authenticate an existing user."""
    username = username.strip()
    if not username or not password:
        return False, "Username and password cannot be empty."
    
    init_db() # Ensure DB is initialized
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return False, "Invalid username or password."
        
        stored_hash = row[0]
        if verify_password(password, stored_hash):
            return True, "Authentication successful."
        else:
            return False, "Invalid username or password."
    except sqlite3.Error as e:
        conn.close()
        return False, f"Database error: {str(e)}"

def save_chat_history(username: str, question: str, answer: str, chunks: list) -> bool:
    """Saves a single Q&A entry with retrieved chunks to the database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        chunks_json = json.dumps(chunks)
        cursor.execute("""
            INSERT INTO chat_history (username, question, answer, chunks_json)
            VALUES (?, ?, ?, ?)
        """, (username, question, answer, chunks_json))
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error:
        conn.close()
        return False

def load_chat_history(username: str) -> list:
    """Loads all chat history entries for a given user from the database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    history = []
    try:
        cursor.execute("""
            SELECT question, answer, chunks_json 
            FROM chat_history 
            WHERE username = ? 
            ORDER BY id ASC
        """, (username,))
        rows = cursor.fetchall()
        conn.close()
        for row in rows:
            try:
                chunks = json.loads(row[2])
            except Exception:
                chunks = []
            history.append({
                "question": row[0],
                "answer": row[1],
                "chunks": chunks
            })
    except sqlite3.Error:
        conn.close()
    return history

def clear_chat_history(username: str) -> bool:
    """Clears all chat history entries for a given user from the database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM chat_history WHERE username = ?", (username,))
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error:
        conn.close()
        return False
