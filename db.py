import mysql.connector

def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="@Hiruni1116*",
        database="student_score"
    )
