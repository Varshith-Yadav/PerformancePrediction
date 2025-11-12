import mysql.connector
import pandas as pd 

def get_connection():
    return mysql.connector.connect(
        host = 'localhost',
        user = 'root',
        password = '*******',
        database = 'students'
    )


def fetch_data():
    conn = get_connection()
    query = ' select * from students order by timestamps'

    df = pd.read_sql(query, conn)
    conn.close()
    return df 

