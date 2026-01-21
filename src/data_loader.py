"""
Модуль для загрузки данных из базы данных PostgreSQL
"""

import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

def connect_to_db():
    """
    Устанавливает соединение с базой данных
    Возвращает объект соединения или None в случае ошибки
    """
    try:
        connection = psycopg2.connect(
            host=os.getenv('DB_HOST', '84.201.134.129'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'skillfactory'),
            user=os.getenv('DB_USER', 'skillfactory'),
            password=os.getenv('DB_PASSWORD', 'cCkxxLVrDE8EbvjueeMedPKt')
        )
        print("✅ Успешное подключение к базе данных")
        return connection
    except Exception as e:
        print(f"❌ Ошибка подключения к базе данных: {e}")
        return None

def load_purchase_data():
    """
    Загружает данные о покупках курсов пользователями
    Возвращает DataFrame с информацией о пользователях и купленных курсах
    """
    query = """
    WITH successful_purchases AS (
        SELECT 
            c.user_id,
            c.id as cart_id,
            ci.resource_id as course_id,
            c.purchased_at,
            c.updated_at
        FROM final.carts c
        JOIN final.cart_items ci ON c.id = ci.cart_id
        WHERE c.state = 'successful' 
          AND ci.resource_type = 'Course'
          AND ci.resource_id IS NOT NULL
          AND c.user_id IS NOT NULL
    ),
    user_course_counts AS (
        SELECT 
            user_id,
            COUNT(DISTINCT course_id) as courses_purchased
        FROM successful_purchases
        GROUP BY user_id
        HAVING COUNT(DISTINCT course_id) > 1
    )
    SELECT 
        sp.user_id,
        sp.course_id,
        sp.purchased_at,
        sp.updated_at
    FROM successful_purchases sp
    JOIN user_course_counts ucc ON sp.user_id = ucc.user_id
    ORDER BY sp.user_id, sp.purchased_at
    """
    
    try:
        conn = connect_to_db()
        if conn is None:
            raise ConnectionError("Не удалось подключиться к базе данных")
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"✅ Загружено {len(df):,} записей о покупках")
        print(f"✅ Уникальных пользователей: {df['user_id'].nunique():,}")
        print(f"✅ Уникальных курсов: {df['course_id'].nunique():,}")
        
        return df
    except Exception as e:
        print(f"❌ Ошибка при загрузке данных: {e}")
        # Возвращаем пустой DataFrame для возможности тестирования
        return pd.DataFrame(columns=['user_id', 'course_id', 'purchased_at', 'updated_at'])

def get_purchase_statistics(df):
    """
    Выводит статистику по загруженным данным о покупках
    """
    if df.empty:
        print("❌ DataFrame пустой")
        return
    
    print("📊 СТАТИСТИКА ПОКУПОК:")
    print(f"   • Всего записей: {len(df):,}")
    print(f"   • Уникальных пользователей: {df['user_id'].nunique():,}")
    print(f"   • Уникальных курсов: {df['course_id'].nunique():,}")
    print(f"   • Период данных: с {df['purchased_at'].min()} по {df['purchased_at'].max()}")
    
    # Количество покупок на пользователя
    purchases_per_user = df.groupby('user_id')['course_id'].nunique()
    print(f"   • Среднее количество курсов на пользователя: {purchases_per_user.mean():.2f}")
    print(f"   • Максимальное количество курсов у одного пользователя: {purchases_per_user.max()}")
    
    return purchases_per_user

if __name__ == "__main__":
    # Тестирование модуля
    print("Тестирование модуля data_loader.py")
    print("=" * 50)
    
    # Загружаем данные
    df = load_purchase_data()
    
    # Выводим статистику
    if not df.empty:
        get_purchase_statistics(df)
        print("\nПервые 5 строк данных:")
        print(df.head())