import sqlite3
from contextlib import contextmanager

class DatabaseManager:
    def __init__(self, db_path='fraud_detection.db'):
        self.db_path = db_path

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def init_database(self):
        """Initialize the database with required tables"""
        with self.get_connection() as conn:
            # Users table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    account_created DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Transactions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    amount REAL NOT NULL,
                    fraud_score REAL NOT NULL,
                    decision TEXT NOT NULL,
                    reason TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')

            conn.commit()

    def get_or_create_user(self, name, email):
        """Get existing user or create new one"""
        with self.get_connection() as conn:
            user = conn.execute(
                'SELECT id FROM users WHERE email = ?', (email,)
            ).fetchone()

            if user:
                return user['id']

            cursor = conn.execute(
                'INSERT INTO users (name, email) VALUES (?, ?)',
                (name, email)
            )
            conn.commit()
            return cursor.lastrowid

    def store_transaction(self, user_id, amount, fraud_score, decision, reason):
        """Store transaction in database"""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                INSERT INTO transactions (user_id, amount, fraud_score, decision, reason)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, amount, fraud_score, decision, reason))
            conn.commit()
            return cursor.lastrowid

    def get_user_transaction_history(self, user_id, days=30):
        """Get user's recent transaction history"""
        with self.get_connection() as conn:
            return conn.execute('''
                SELECT * FROM transactions 
                WHERE user_id = ? AND created_at >= datetime('now', '-{} days')
                ORDER BY created_at DESC
            '''.format(days), (user_id,)).fetchall()

    def get_user_transaction_speed(self, user_id, minutes=60):
        """Count transactions in last X minutes"""
        with self.get_connection() as conn:
            result = conn.execute('''
                SELECT COUNT(*) as count FROM transactions 
                WHERE user_id = ? AND created_at >= datetime('now', '-{} minutes')
            '''.format(minutes), (user_id,)).fetchone()
            return result['count'] if result else 0

    def get_past_transaction_count(self, user_id, days=30):
        """Get number of transactions in the last N hours"""
        with self.get_connection() as conn:
            result = conn.execute(f'''
                SELECT COUNT(*) as count FROM transactions
            
                WHERE user_id = ? AND created_at >= datetime('now', '-{days} days') 
            ''', (user_id,)).fetchone()
            return result['count'] if result else 0

    def get_account_age_days(self, user_id):
        """Get account age in days"""
        with self.get_connection() as conn:
            result = conn.execute('''
                SELECT julianday('now') - julianday(account_created) as age_days
                FROM users WHERE id = ?
            ''', (user_id,)).fetchone()
            return result['age_days'] if result else 0

    def get_dashboard_stats(self):
        """Get basic dashboard statistics"""
        with self.get_connection() as conn:
            total_transactions = conn.execute(
                'SELECT COUNT(*) as count FROM transactions'
            ).fetchone()['count']
            fraud_transactions = conn.execute(
                'SELECT COUNT(*) as count FROM transactions WHERE decision = "Under Review"'
            ).fetchone()['count']

            fraud_rate = (fraud_transactions / total_transactions * 100) if total_transactions > 0 else 0

            return {
                'total_transactions': total_transactions,
                'fraud_transactions': fraud_transactions,
                'fraud_rate': round(fraud_rate, 2)
            }

    def get_dashboard_data(self):
        """Get comprehensive dashboard data"""
        with self.get_connection() as conn:
            # Daily fraud rate for last 30 days
            daily_data = conn.execute('''
                SELECT 
                    DATE(created_at) as date,
                    COUNT(*) as total,
                    SUM(CASE WHEN decision = "Under Review" THEN 1 ELSE 0 END) as fraud
                FROM transactions 
                WHERE created_at >= date('now', '-30 days')
                GROUP BY DATE(created_at)
                ORDER BY date
            ''').fetchall()

            # Common fraud reasons
            fraud_reasons = conn.execute('''
                SELECT reason, COUNT(*) as count
                FROM transactions 
                WHERE decision = "Under Review" AND reason IS NOT NULL
                GROUP BY reason
                ORDER BY count DESC
                LIMIT 10
            ''').fetchall()

            # Recent transactions
            recent_transactions = conn.execute('''
                SELECT t.*, u.name, u.email
                FROM transactions t
                JOIN users u ON t.user_id = u.id
                ORDER BY t.created_at DESC
                LIMIT 50
            ''').fetchall()

            daily_data_list = [{
                'date': row['date'] or '',
                'total': row['total'] or 0,
                'fraud': row['fraud'] or 0
            } for row in daily_data]

            fraud_reasons_list = [{
                'reason': row['reason'],
                'count': row['count'] or 0
            } for row in fraud_reasons if row['reason']]

            recent_transactions_list = [{
                'id': row['id'],
                'user_id': row['user_id'],
                'amount': float(row['amount']) if row['amount'] else 0.0,
                'fraud_score': int(row['fraud_score']) if row['fraud_score'] else 0,
                'decision': row['decision'] or 'Unknown',
                'reason': row['reason'] or 'No reason provided',
                'created_at': row['created_at'] or '',
                'name': row['name'] or 'Unknown',
                'email': row['email'] or 'Unknown'
            } for row in recent_transactions]

            return {
                'daily_fraud_rate': daily_data_list,
                'fraud_reasons': fraud_reasons_list,
                'recent_transactions': recent_transactions_list
            }
