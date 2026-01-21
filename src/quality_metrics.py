"""
Модуль для создания и анализа метрик качества курсов
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def create_synthetic_quality_metrics(course_ids):
    """
    Создает синтетические метрики качества для курсов
    
    Parameters:
    -----------
    course_ids : list
        Список ID курсов
    
    Returns:
    --------
    DataFrame : Метрики качества для каждого курса
    """
    np.random.seed(42)
    n_courses = len(course_ids)
    
    print(f"📊 Создание синтетических метрик качества для {n_courses} курсов...")
    
    # Создаем DataFrame с метриками качества
    quality_metrics = pd.DataFrame({
        'course_id': course_ids,
        'course_name': [f'Курс {i}' for i in course_ids],
        
        # COR (Completion Rate) - процент студентов, завершивших курс
        'cor': np.random.beta(5, 2, n_courses) * 40 + 30,  # 30-70%
        
        # CSI (Customer Satisfaction Index) - удовлетворенность
        'csi': np.random.beta(8, 2, n_courses) * 2 + 3,  # 3-5 баллов
        
        # NPS (Net Promoter Score) - лояльность
        'nps': np.random.normal(20, 30, n_courses),  # -50 до 80
        
        # Среднее время проверки ДЗ (в часах)
        'homework_check_time': np.random.exponential(24, n_courses),
        
        # Retention rate - вероятность покупки следующего курса
        'retention_rate': np.random.beta(3, 5, n_courses) * 40 + 10,  # 10-50%
        
        # Процент положительных отзывов
        'positive_reviews': np.random.beta(8, 2, n_courses) * 40 + 40,  # 40-80%
        
        # Рейтинг преподавателя
        'teacher_rating': np.random.beta(9, 2, n_courses) * 2 + 3,  # 3-5 баллов
    })
    
    # Ограничиваем значения в разумных пределах
    quality_metrics['nps'] = quality_metrics['nps'].clip(-100, 100)
    quality_metrics['homework_check_time'] = quality_metrics['homework_check_time'].clip(1, 168)
    quality_metrics['retention_rate'] = quality_metrics['retention_rate'].clip(5, 80)
    
    # Рассчитываем интегральный показатель качества
    quality_metrics = calculate_quality_score(quality_metrics)
    
    print("✅ Метрики качества успешно созданы")
    return quality_metrics

def calculate_quality_score(quality_metrics):
    """
    Рассчитывает интегральный показатель качества курса
    """
    # Создаем временные нормализованные значения
    quality_metrics['cor_norm'] = quality_metrics['cor'] / 100
    quality_metrics['csi_norm'] = (quality_metrics['csi'] - 1) / 4  # 1-5 -> 0-1
    quality_metrics['nps_norm'] = (quality_metrics['nps'] + 100) / 200  # -100..100 -> 0-1
    quality_metrics['hw_norm'] = 1 - (quality_metrics['homework_check_time'].clip(1, 72) / 72)
    quality_metrics['retention_norm'] = quality_metrics['retention_rate'] / 100
    quality_metrics['reviews_norm'] = quality_metrics['positive_reviews'] / 100
    quality_metrics['teacher_norm'] = (quality_metrics['teacher_rating'] - 1) / 4
    
    # Веса для каждой метрики
    weights = {
        'cor_norm': 0.25,
        'csi_norm': 0.20,
        'nps_norm': 0.15,
        'hw_norm': 0.10,
        'retention_norm': 0.15,
        'reviews_norm': 0.10,
        'teacher_norm': 0.05
    }
    
    # Рассчитываем итоговый score
    quality_metrics['quality_score'] = 0
    for col, weight in weights.items():
        quality_metrics['quality_score'] += quality_metrics[col] * weight
    
    # Масштабируем до 0-100
    quality_metrics['quality_score'] = quality_metrics['quality_score'] * 100
    
    # Добавляем категорию качества
    quality_metrics['quality_category'] = quality_metrics['quality_score'].apply(
        lambda x: 'Высокое' if x >= 70 else 'Среднее' if x >= 50 else 'Низкое'
    )
    
    # Удаляем временные столбцы
    cols_to_drop = [col for col in quality_metrics.columns if col.endswith('_norm')]
    quality_metrics = quality_metrics.drop(columns=cols_to_drop)
    
    return quality_metrics

def analyze_quality_distribution(quality_metrics):
    """
    Анализирует распределение качества курсов
    """
    print("\n📈 АНАЛИЗ КАЧЕСТВА КУРСОВ:")
    print("-" * 40)
    
    # Распределение по категориям
    category_counts = quality_metrics['quality_category'].value_counts()
    for category, count in category_counts.items():
        percentage = count / len(quality_metrics) * 100
        print(f"   • {category}: {count} курсов ({percentage:.1f}%)")
    
    # Статистика по quality_score
    print(f"\n   • Средний quality_score: {quality_metrics['quality_score'].mean():.1f}")
    print(f"   • Медианный quality_score: {quality_metrics['quality_score'].median():.1f}")
    print(f"   • Min quality_score: {quality_metrics['quality_score'].min():.1f}")
    print(f"   • Max quality_score: {quality_metrics['quality_score'].max():.1f}")
    
    # Корреляции
    correlation_cols = ['cor', 'csi', 'nps', 'retention_rate', 'quality_score']
    correlation_matrix = quality_metrics[correlation_cols].corr()
    quality_correlation = correlation_matrix.loc['quality_score', 'retention_rate']
    print(f"\n   • Корреляция quality_score с retention_rate: {quality_correlation:.2f}")
    
    return category_counts, correlation_matrix

def plot_quality_distribution(quality_metrics, save_path=None):
    """
    Создает визуализацию распределения качества курсов
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Распределение COR
    axes[0, 0].hist(quality_metrics['cor'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Completion Rate (COR)')
    axes[0, 0].set_xlabel('Процент завершивших курс')
    axes[0, 0].set_ylabel('Количество курсов')
    
    # 2. Распределение CSI
    axes[0, 1].hist(quality_metrics['csi'], bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[0, 1].set_title('Customer Satisfaction Index (CSI)')
    axes[0, 1].set_xlabel('Удовлетворенность (1-5)')
    
    # 3. Распределение NPS
    axes[0, 2].hist(quality_metrics['nps'], bins=20, edgecolor='black', alpha=0.7, color='salmon')
    axes[0, 2].set_title('Net Promoter Score (NPS)')
    axes[0, 2].set_xlabel('NPS (-100 до 100)')
    
    # 4. Распределение Retention Rate
    axes[1, 0].hist(quality_metrics['retention_rate'], bins=20, edgecolor='black', alpha=0.7, color='gold')
    axes[1, 0].set_title('Retention Rate')
    axes[1, 0].set_xlabel('Процент повторных покупок')
    axes[1, 0].set_ylabel('Количество курсов')
    
    # 5. Распределение времени проверки ДЗ
    axes[1, 1].hist(quality_metrics['homework_check_time'], bins=20, edgecolor='black', alpha=0.7, color='violet')
    axes[1, 1].set_title('Время проверки домашних заданий')
    axes[1, 1].set_xlabel('Часы')
    
    # 6. Распределение интегрального качества
    colors = ['red' if x < 50 else 'orange' if x < 70 else 'green' for x in quality_metrics['quality_score']]
    axes[1, 2].bar(range(len(quality_metrics)), sorted(quality_metrics['quality_score']), color=colors)
    axes[1, 2].set_title('Интегральный показатель качества')
    axes[1, 2].set_xlabel('Курсы (отсортированы)')
    axes[1, 2].set_ylabel('Quality Score')
    
    plt.suptitle('Распределение метрик качества курсов', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 График сохранен: {save_path}")
    
    plt.show()
    
    return fig

def get_courses_by_quality(quality_metrics, category):
    """
    Возвращает список курсов по категории качества
    """
    if category not in ['Высокое', 'Среднее', 'Низкое']:
        raise ValueError("Категория должна быть: 'Высокое', 'Среднее' или 'Низкое'")
    
    filtered = quality_metrics[quality_metrics['quality_category'] == category]
    return filtered['course_id'].tolist()

if __name__ == "__main__":
    # Тестирование модуля
    print("Тестирование модуля quality_metrics.py")
    print("=" * 50)
    
    # Создаем тестовые ID курсов
    test_course_ids = list(range(1, 101))
    
    # Создаем метрики качества
    quality_df = create_synthetic_quality_metrics(test_course_ids)
    
    # Анализируем распределение
    analyze_quality_distribution(quality_df)
    
    # Визуализируем
    plot_quality_distribution(quality_df)
    
    # Пример фильтрации
    high_quality_courses = get_courses_by_quality(quality_df, 'Высокое')
    print(f"\nКурсы высокого качества: {len(high_quality_courses)}")