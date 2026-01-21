"""
Модуль рекомендательной системы курсов
"""

import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter

class CourseRecommender:
    """Рекомендательная система с учетом качества курсов"""
    
    def __init__(self, pair_counts, quality_metrics, all_courses, threshold=9):
        """
        Инициализация рекомендательной системы
        
        Parameters:
        -----------
        pair_counts : Counter
            Счетчик пар курсов с частотами
        quality_metrics : DataFrame
            Метрики качества курсов
        all_courses : array
            Все уникальные курсы
        threshold : int
            Минимальная частота для учета пары
        """
        self.pair_counts = {pair: count for pair, count in pair_counts.items() if count > threshold}
        self.quality_metrics = quality_metrics.set_index('course_id')
        self.all_courses = set(all_courses)
        self.threshold = threshold
        
        # Создаем индекс рекомендаций
        self.recommendation_index = self._build_recommendation_index()
        
        print(f"🤖 Рекомендательная система создана:")
        print(f"   • Всего курсов в системе: {len(self.all_courses)}")
        print(f"   • Курсов в частых парах: {len(self.recommendation_index)}")
        print(f"   • Курсов без частых пар: {len(self.all_courses) - len(self.recommendation_index)}")
    
    def _build_recommendation_index(self):
        """Создает индекс рекомендаций для каждого курса"""
        index = {}
        for (course1, course2), count in self.pair_counts.items():
            # Получаем качество курсов
            score1 = self._get_course_score(course1)
            score2 = self._get_course_score(course2)
            
            # Комбинированный вес: частота * среднее качество
            weight = count * ((score1 + score2) / 2)
            
            if course1 not in index:
                index[course1] = []
            index[course1].append((course2, weight))
            
            if course2 not in index:
                index[course2] = []
            index[course2].append((course1, weight))
        
        # Сортируем рекомендации по весу (убывающая)
        for course in index:
            index[course].sort(key=lambda x: x[1], reverse=True)
            
        return index
    
    def _get_course_score(self, course_id):
        """Получает интегральный показатель качества курса"""
        if course_id in self.quality_metrics.index:
            return self.quality_metrics.loc[course_id, 'quality_score'] / 100  # Нормализуем до 0-1
        return 0.5  # Значение по умолчанию
    
    def get_recommendations(self, course_id, n=2, min_quality=50):
        """
        Получает рекомендации для заданного курса
        
        Parameters:
        -----------
        course_id : int
            ID курса для которого нужны рекомендации
        n : int
            Количество рекомендаций для возврата
        min_quality : int
            Минимальный quality_score для рекомендации
            
        Returns:
        --------
        list : Список рекомендуемых курсов
        """
        if course_id not in self.recommendation_index:
            return []
        
        recommendations = []
        for candidate_course, weight in self.recommendation_index[course_id]:
            # Проверяем качество кандидата
            candidate_score = self._get_course_score(candidate_course) * 100
            
            if candidate_score >= min_quality:
                recommendations.append(candidate_course)
            
            if len(recommendations) >= n:
                break
        
        return recommendations
    
    def get_all_recommendations(self, n=2, min_quality=50):
        """
        Получает рекомендации для всех курсов
        
        Returns:
        --------
        DataFrame : Таблица с рекомендациями
        """
        recommendations = []
        
        for course_id in sorted(self.all_courses):
            recs = self.get_recommendations(course_id, n, min_quality)
            
            # Если рекомендаций меньше, чем нужно, ищем курсы с высоким качеством
            if len(recs) < n:
                # Ищем курсы с высоким качеством, отличные от текущего
                if hasattr(self.quality_metrics, 'index'):
                    high_quality_courses = self.quality_metrics[
                        (self.quality_metrics['quality_score'] >= min_quality) &
                        (self.quality_metrics.index != course_id)
                    ].index.tolist()
                else:
                    high_quality_courses = []
                
                # Исключаем уже рекомендованные
                available = [c for c in high_quality_courses if c not in recs]
                
                # Добираем нужное количество
                while len(recs) < n and available:
                    new_rec = np.random.choice(available)
                    recs.append(new_rec)
                    available.remove(new_rec)
            
            # Заполняем None, если все еще не хватает
            while len(recs) < n:
                recs.append(None)
            
            # Получаем качество рекомендаций
            rec_quality = []
            for rec in recs:
                if rec is not None and rec in self.quality_metrics.index:
                    rec_quality.append(self.quality_metrics.loc[rec, 'quality_score'])
                else:
                    rec_quality.append(None)
            
            recommendations.append({
                'course_id': course_id,
                'course_quality': self._get_course_score(course_id) * 100,
                'recomm_one': recs[0],
                'recomm_one_quality': rec_quality[0],
                'recomm_two': recs[1],
                'recomm_two_quality': rec_quality[1],
                'has_recommendations': recs[0] is not None
            })
        
        return pd.DataFrame(recommendations)
    
    def get_recommendation_statistics(self, recommendations_df):
        """
        Выводит статистику рекомендаций
        """
        print("\n📊 СТАТИСТИКА РЕКОМЕНДАЦИЙ:")
        print("-" * 40)
        
        total_courses = len(recommendations_df)
        courses_with_recs = recommendations_df['has_recommendations'].sum()
        
        print(f"   • Всего курсов: {total_courses}")
        print(f"   • Курсов с рекомендациями: {courses_with_recs} ({courses_with_recs/total_courses*100:.1f}%)")
        
        # Среднее качество рекомендаций
        quality_cols = ['recomm_one_quality', 'recomm_two_quality']
        avg_quality = recommendations_df[quality_cols].mean().mean()
        print(f"   • Среднее качество рекомендаций: {avg_quality:.1f}")
        
        # Распределение по количеству рекомендаций
        rec_counts = recommendations_df[['recomm_one', 'recomm_two']].notna().sum(axis=1)
        for count in [0, 1, 2]:
            count_courses = (rec_counts == count).sum()
            print(f"   • Курсов с {count} рекомендациями: {count_courses} ({count_courses/total_courses*100:.1f}%)")
        
        return {
            'total_courses': total_courses,
            'courses_with_recs': courses_with_recs,
            'avg_quality': avg_quality
        }

def analyze_joint_purchases(purchase_data):
    """
    Анализирует совместные покупки курсов
    
    Parameters:
    -----------
    purchase_data : DataFrame
        Данные о покупках курсов
        
    Returns:
    --------
    Counter : Счетчик пар курсов
    DataFrame : Статистика пар
    """
    print("🔄 Анализ совместных покупок...")
    
    # Создаем список курсов для каждого пользователя
    user_courses = purchase_data.groupby('user_id')['course_id'].apply(list).reset_index()
    
    # Создаем все возможные пары курсов для каждого пользователя
    all_pairs = []
    for courses in user_courses['course_id']:
        if len(courses) >= 2:
            pairs = list(combinations(sorted(courses), 2))
            all_pairs.extend(pairs)
    
    # Подсчитываем частоту пар
    pair_counts = Counter(all_pairs)
    
    print(f"   • Всего уникальных пар курсов: {len(pair_counts):,}")
    print(f"   • Всего совместных покупок: {len(all_pairs):,}")
    
    # Создаем DataFrame для анализа
    pair_freq_df = pd.DataFrame(pair_counts.most_common(), columns=['pair', 'frequency'])
    pair_freq_df['course1'] = pair_freq_df['pair'].apply(lambda x: x[0])
    pair_freq_df['course2'] = pair_freq_df['pair'].apply(lambda x: x[1])
    
    return pair_counts, pair_freq_df

def print_top_pairs(pair_counts, n=10):
    """
    Выводит топ-N самых популярных пар курсов
    """
    print(f"\n🏆 Топ-{n} самых популярных пар курсов:")
    for i, (pair, freq) in enumerate(pair_counts.most_common(n)):
        print(f"   {i+1}. Курсы {pair[0]} и {pair[1]}: {freq} совместных покупок")

def get_recommendations_for_course(recommendations_df, course_id):
    """
    Получает рекомендации для конкретного курса
    
    Parameters:
    -----------
    recommendations_df : DataFrame
        Таблица рекомендаций
    course_id : int
        ID курса
        
    Returns:
    --------
    dict : Информация о рекомендациях
    """
    rec_row = recommendations_df[recommendations_df['course_id'] == course_id]
    
    if rec_row.empty:
        return {"error": f"Курс {course_id} не найден"}
    
    rec_row = rec_row.iloc[0]
    
    result = {
        'course_id': course_id,
        'course_quality': rec_row['course_quality'],
        'recommendations': []
    }
    
    if pd.notna(rec_row['recomm_one']):
        result['recommendations'].append({
            'course_id': rec_row['recomm_one'],
            'quality_score': rec_row['recomm_one_quality']
        })
    
    if pd.notna(rec_row['recomm_two']):
        result['recommendations'].append({
            'course_id': rec_row['recomm_two'],
            'quality_score': rec_row['recomm_two_quality']
        })
    
    return result

if __name__ == "__main__":
    # Тестирование модуля
    print("Тестирование модуля recommender.py")
    print("=" * 50)
    
    # Создаем тестовые данные
    test_purchases = pd.DataFrame({
        'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        'course_id': [101, 102, 101, 103, 102, 103, 101, 104, 102, 104]
    })
    
    test_quality = pd.DataFrame({
        'course_id': [101, 102, 103, 104],
        'quality_score': [85, 72, 65, 90]
    })
    
    # Анализируем совместные покупки
    pair_counts, pair_df = analyze_joint_purchases(test_purchases)
    
    # Создаем рекомендательную систему
    all_courses = test_purchases['course_id'].unique()
    recommender = CourseRecommender(pair_counts, test_quality, all_courses, threshold=1)
    
    # Получаем рекомендации
    recommendations = recommender.get_all_recommendations(n=2, min_quality=60)
    
    # Выводим статистику
    recommender.get_recommendation_statistics(recommendations)
    
    # Пример получения рекомендаций для конкретного курса
    course_id = 101
    recs = get_recommendations_for_course(recommendations, course_id)
    print(f"\n📋 Рекомендации для курса {course_id}:")
    for i, rec in enumerate(recs.get('recommendations', []), 1):
        print(f"   {i}. Курс {rec['course_id']} (качество: {rec['quality_score']})")