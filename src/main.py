"""
Основной скрипт для запуска полного анализа рекомендательной системы
"""

import os
import sys
from datetime import datetime

# Добавляем src в путь импорта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_purchase_data, get_purchase_statistics
from src.quality_metrics import create_synthetic_quality_metrics, analyze_quality_distribution
from src.recommender import (
    analyze_joint_purchases, 
    print_top_pairs, 
    CourseRecommender,
    get_recommendations_for_course
)
from src.ltv_calculator import (
    calculate_ltv_scenarios,
    simulate_ab_test,
    plot_ltv_comparison,
    plot_ab_test_results,
    calculate_roi
)

def save_results(output_folder="results"):
    """
    Сохраняет все результаты анализа в указанную папку
    """
    os.makedirs(output_folder, exist_ok=True)
    print(f"📁 Результаты будут сохранены в папку: {output_folder}")
    return output_folder

def main():
    """Основная функция анализа"""
    print("=" * 80)
    print("🚀 ЗАПУСК АНАЛИЗА РЕКОМЕНДАТЕЛЬНОЙ СИСТЕМЫ")
    print("=" * 80)
    
    # Создаем папку для результатов
    output_folder = save_results("analysis_results")
    
    # 1. Загрузка данных
    print("\n📥 1. ЗАГРУЗКА ДАННЫХ О ПОКУПКАХ")
    df_purchases = load_purchase_data()
    
    if df_purchases.empty:
        print("❌ Не удалось загрузить данные. Завершение работы.")
        return
    
    get_purchase_statistics(df_purchases)
    
    # 2. Анализ совместных покупок
    print("\n🔄 2. АНАЛИЗ СОВМЕСТНЫХ ПОКУПОК")
    pair_counts, pair_df = analyze_joint_purchases(df_purchases)
    print_top_pairs(pair_counts, n=10)
    
    # Сохраняем статистику пар
    pair_df.to_csv(os.path.join(output_folder, 'course_pair_statistics.csv'), index=False)
    print(f"✅ Статистика пар сохранена: {output_folder}/course_pair_statistics.csv")
    
    # 3. Создание метрик качества
    print("\n🎯 3. СОЗДАНИЕ МЕТРИК КАЧЕСТВА КУРСОВ")
    all_courses = df_purchases['course_id'].unique()
    quality_metrics = create_synthetic_quality_metrics(all_courses)
    analyze_quality_distribution(quality_metrics)
    
    # Сохраняем метрики качества
    quality_metrics.to_csv(os.path.join(output_folder, 'course_quality_metrics.csv'), index=False)
    print(f"✅ Метрики качества сохранены: {output_folder}/course_quality_metrics.csv")
    
    # 4. Создание рекомендательной системы
    print("\n🤖 4. СОЗДАНИЕ РЕКОМЕНДАТЕЛЬНОЙ СИСТЕМЫ")
    recommender = CourseRecommender(pair_counts, quality_metrics, all_courses, threshold=9)
    
    # Получаем рекомендации без учета качества
    print("\n   a) Рекомендации без учета качества:")
    recommendations_no_quality = recommender.get_all_recommendations(min_quality=0)
    stats_no_quality = recommender.get_recommendation_statistics(recommendations_no_quality)
    
    # Получаем рекомендации с учетом качества
    print("\n   b) Рекомендации с учетом качества (min_quality=60):")
    recommendations_with_quality = recommender.get_all_recommendations(min_quality=60)
    stats_with_quality = recommender.get_recommendation_statistics(recommendations_with_quality)
    
    # Сохраняем рекомендации
    final_recommendations = recommendations_with_quality[['course_id', 'recomm_one', 'recomm_two']]
    final_recommendations.to_csv(os.path.join(output_folder, 'final_course_recommendations.csv'), index=False)
    print(f"✅ Рекомендации сохранены: {output_folder}/final_course_recommendations.csv")
    
    # 5. Расчет LTV
    print("\n💰 5. РАСЧЕТ LIFETIME VALUE (LTV)")
    ltv_comparison = calculate_ltv_scenarios()
    print(ltv_comparison.to_string(index=False))
    
    # Сохраняем результаты LTV
    ltv_comparison.to_csv(os.path.join(output_folder, 'ltv_analysis_results.csv'), index=False)
    print(f"✅ Результаты LTV сохранены: {output_folder}/ltv_analysis_results.csv")
    
    # Визуализация LTV
    plot_ltv_comparison(ltv_comparison, save_path=os.path.join(output_folder, 'ltv_comparison.png'))
    
    # 6. A/B-тестирование
    print("\n🔬 6. A/B-ТЕСТИРОВАНИЕ РЕКОМЕНДАЦИЙ")
    ab_test_results = simulate_ab_test()
    print(ab_test_results.to_string(index=False))
    
    # Сохраняем результаты A/B-теста
    ab_test_results.to_csv(os.path.join(output_folder, 'ab_test_results.csv'), index=False)
    print(f"✅ Результаты A/B-теста сохранены: {output_folder}/ab_test_results.csv")
    
    # Визуализация A/B-теста
    plot_ab_test_results(ab_test_results, save_path=os.path.join(output_folder, 'ab_test_results.png'))
    
    # 7. Расчет ROI
    print("\n📈 7. РАСЧЕТ ROI (RETURN ON INVESTMENT)")
    roi_results = calculate_roi(
        development_cost=500000,
        monthly_maintenance=50000,
        monthly_revenue_increase=1000000 * 0.17,
        months=12
    )
    
    print(f"   • ROI за год: {roi_results['roi_percent']:.1f}%")
    print(f"   • Срок окупаемости: {roi_results['payback_months']:.1f} месяцев")
    print(f"   • Чистая прибыль за год: {roi_results['net_profit']:,.0f} руб.")
    
    # 8. Сводный отчет
    print("\n📋 8. ИТОГОВЫЙ ОТЧЕТ")
    summary_report = f"""
    ИТОГОВЫЙ ОТЧЕТ ПО РЕКОМЕНДАТЕЛЬНОЙ СИСТЕМЕ
    Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    КЛЮЧЕВЫЕ РЕЗУЛЬТАТЫ:
    1. Всего проанализировано курсов: {len(all_courses)}
    2. Рост конверсии от рекомендаций: +17%
    3. LTV с учетом качества: +{((ltv_comparison.iloc[2]['LTV (руб.)'] / ltv_comparison.iloc[0]['LTV (руб.)'] - 1) * 100):.1f}%
    4. ROI рекомендательной системы: {roi_results['roi_percent']:.1f}%
    5. Срок окупаемости: {roi_results['payback_months']:.1f} месяцев
    
    РЕКОМЕНДАЦИИ:
    1. Внедрить рекомендательную систему немедленно
    2. Начать сбор реальных метрик качества курсов
    3. Провести аудит курсов с низким качеством
    4. Мониторить эффективность через A/B-тесты
    """
    
    print(summary_report)
    
    # Сохраняем сводный отчет
    with open(os.path.join(output_folder, 'summary_report.txt'), 'w', encoding='utf-8') as f:
        f.write(summary_report)
    print(f"✅ Сводный отчет сохранен: {output_folder}/summary_report.txt")
    
    print("=" * 80)
    print("🎉 АНАЛИЗ УСПЕШНО ЗАВЕРШЕН!")
    print(f"📁 Все файлы сохранены в папке: {os.path.abspath(output_folder)}")
    print("=" * 80)

if __name__ == "__main__":
    main()