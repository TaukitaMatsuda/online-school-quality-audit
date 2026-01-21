"""
Модуль для расчета Lifetime Value (LTV) и A/B тестирования
"""

import numpy as np
import pandas as pd
import math
from scipy import stats
import matplotlib.pyplot as plt

def calculate_ltv(avg_purchase_value, purchase_frequency, customer_lifespan, 
                  retention_rate=None, discount_rate=0.1):
    """
    Расчет Lifetime Value
    
    Parameters:
    -----------
    avg_purchase_value : float
        Средняя стоимость покупки
    purchase_frequency : float
        Частота покупок в год
    customer_lifespan : float
        Ожидаемая продолжительность "жизни" клиента в годах
    retention_rate : float or None
        Годовая retention rate (если None, используется простая модель)
    discount_rate : float
        Ставка дисконтирования
        
    Returns:
    --------
    float : LTV
    """
    if retention_rate is None:
        # Простая модель: LTV = Avg Purchase Value * Purchase Frequency * Customer Lifespan
        ltv = avg_purchase_value * purchase_frequency * customer_lifespan
    else:
        # Модель с учетом retention rate и дисконтирования
        ltv = 0
        for year in range(1, int(customer_lifespan) + 1):
            year_value = avg_purchase_value * purchase_frequency * (retention_rate / 100) ** (year - 1)
            discounted_value = year_value / ((1 + discount_rate) ** (year - 1))
            ltv += discounted_value
    
    return ltv

def simulate_ab_test(n_users=10000, conversion_baseline=0.0335, 
                     effect_recommendations=0.0392, effect_quality=0.045):
    """
    Симуляция A/B-теста рекомендаций с учетом качества курсов
    
    Parameters:
    -----------
    n_users : int
        Общее количество пользователей в тесте
    conversion_baseline : float
        Базовая конверсия (без рекомендаций)
    effect_recommendations : float
        Конверсия с рекомендациями без учета качества
    effect_quality : float
        Конверсия с рекомендациями с учетом качества
        
    Returns:
    --------
    DataFrame : Результаты A/B-теста
    """
    np.random.seed(42)
    
    # Группа A: Без рекомендаций
    group_a_conversion = np.random.binomial(1, conversion_baseline, n_users//2)
    
    # Группа B: С рекомендациями (без учета качества)
    group_b_conversion = np.random.binomial(1, effect_recommendations, n_users//2)
    
    # Группа C: С рекомендациями + качество
    group_c_conversion = np.random.binomial(1, effect_quality, n_users//2)
    
    # Результаты
    results = {
        'Группа': ['A: Без рекомендаций', 'B: Рекомендации', 'C: Рекомендации + качество'],
        'Пользователей': [n_users//2, n_users//2, n_users//2],
        'Конверсия': [
            group_a_conversion.mean(),
            group_b_conversion.mean(),
            group_c_conversion.mean()
        ]
    }
    
    # Статистическая значимость
    from scipy.stats import chi2_contingency
    
    # Сравнение A vs B
    contingency_ab = [[sum(group_a_conversion), len(group_a_conversion) - sum(group_a_conversion)],
                     [sum(group_b_conversion), len(group_b_conversion) - sum(group_b_conversion)]]
    chi2_ab, p_ab, _, _ = chi2_contingency(contingency_ab)
    
    # Сравнение B vs C
    contingency_bc = [[sum(group_b_conversion), len(group_b_conversion) - sum(group_b_conversion)],
                     [sum(group_c_conversion), len(group_c_conversion) - sum(group_c_conversion)]]
    chi2_bc, p_bc, _, _ = chi2_contingency(contingency_bc)
    
    results['p-value (vs A)'] = ['-', p_ab, p_bc]
    results['Стат. значимость'] = ['-', 'ДА' if p_ab < 0.05 else 'НЕТ', 'ДА' if p_bc < 0.05 else 'НЕТ']
    
    return pd.DataFrame(results)

def calculate_ltv_scenarios(avg_course_price=15000, avg_customer_lifespan=2, discount_rate=0.1):
    """
    Расчет LTV для разных сценариев
    
    Returns:
    --------
    DataFrame : Сравнение LTV по сценариям
    """
    # Сценарий 1: Базовый (без рекомендательной системы)
    ltv_baseline = calculate_ltv(
        avg_purchase_value=avg_course_price,
        purchase_frequency=1.0,
        customer_lifespan=avg_customer_lifespan,
        retention_rate=20,
        discount_rate=discount_rate
    )
    
    # Сценарий 2: С рекомендательной системой (без учета качества)
    ltv_recommendations = calculate_ltv(
        avg_purchase_value=avg_course_price * 1.17,
        purchase_frequency=1.2,
        customer_lifespan=avg_customer_lifespan,
        retention_rate=20,
        discount_rate=discount_rate
    )
    
    # Сценарий 3: С рекомендательной системой + только качественные курсы
    ltv_quality_recommendations = calculate_ltv(
        avg_purchase_value=avg_course_price * 1.15,
        purchase_frequency=1.15,
        customer_lifespan=avg_customer_lifespan * 1.5,
        retention_rate=35,
        discount_rate=discount_rate
    )
    
    # Сценарий 4: Идеальный (качественные курсы + персонализация)
    ltv_ideal = calculate_ltv(
        avg_purchase_value=avg_course_price * 1.25,
        purchase_frequency=1.3,
        customer_lifespan=avg_customer_lifespan * 2,
        retention_rate=50,
        discount_rate=discount_rate
    )
    
    # Создаем таблицу сравнения
    ltv_comparison = pd.DataFrame({
        'Сценарий': [
            'Базовый (без рекомендаций)',
            'Рекомендации без учета качества',
            'Рекомендации + качество курсов',
            'Идеальный (качество + персонализация)'
        ],
        'LTV (руб.)': [ltv_baseline, ltv_recommendations, ltv_quality_recommendations, ltv_ideal],
        'Рост vs базовый': ['-', 
                           f'+{(ltv_recommendations/ltv_baseline-1)*100:.1f}%',
                           f'+{(ltv_quality_recommendations/ltv_baseline-1)*100:.1f}%',
                           f'+{(ltv_ideal/ltv_baseline-1)*100:.1f}%'],
        'Средний чек': [avg_course_price, 
                       avg_course_price * 1.17,
                       avg_course_price * 1.15,
                       avg_course_price * 1.25],
        'Retention rate': ['20%', '20%', '35%', '50%'],
        'Срок жизни (лет)': [avg_customer_lifespan, 
                            avg_customer_lifespan,
                            avg_customer_lifespan * 1.5,
                            avg_customer_lifespan * 2]
    })
    
    return ltv_comparison

def plot_ltv_comparison(ltv_comparison, save_path=None):
    """
    Визуализация сравнения LTV по сценариям
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Сравнение LTV по сценариям
    scenarios = ltv_comparison['Сценарий']
    ltv_values = ltv_comparison['LTV (руб.)'] / 1000  # В тысячах рублей
    
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
    bars = axes[0].bar(scenarios, ltv_values, color=colors, edgecolor='black')
    axes[0].set_title('Lifetime Value по сценариям', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('LTV (тыс. руб.)', fontsize=11)
    axes[0].set_xticklabels(scenarios, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    for bar, value in zip(bars, ltv_values):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, height + 2,
                    f'{value:,.0f}K', ha='center', va='bottom', fontweight='bold')
    
    # 2. Рост LTV vs базовый сценарий
    growth_values = [(v/ltv_values[0]-1)*100 for v in ltv_values][1:]  # Пропускаем базовый
    growth_scenarios = scenarios[1:]
    
    axes[1].bar(growth_scenarios, growth_values, color=['#f39c12', '#2ecc71', '#3498db'], 
                edgecolor='black')
    axes[1].set_title('Рост LTV относительно базового сценария', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Рост LTV, %', fontsize=11)
    axes[1].set_xticklabels(growth_scenarios, rotation=45, ha='right')
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения
    for i, value in enumerate(growth_values):
        axes[1].text(i, value + 2 if value > 0 else value - 10, 
                    f'+{value:.1f}%' if value > 0 else f'{value:.1f}%',
                    ha='center', va='bottom' if value > 0 else 'top',
                    fontweight='bold')
    
    plt.suptitle('Анализ LTV (Lifetime Value)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 График LTV сохранен: {save_path}")
    
    plt.show()
    
    return fig

def plot_ab_test_results(ab_test_results, save_path=None):
    """
    Визуализация результатов A/B-теста
    """
    plt.figure(figsize=(10, 6))
    groups = ab_test_results['Группа']
    conversion_rates = ab_test_results['Конверсия'] * 100
    
    bars = plt.bar(groups, conversion_rates, color=['#e74c3c', '#f39c12', '#2ecc71'], 
                   edgecolor='black', alpha=0.8)
    plt.title('Результаты A/B-теста рекомендаций', fontsize=14, fontweight='bold')
    plt.ylabel('Конверсия, %', fontsize=12)
    plt.xlabel('Группа теста', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    for bar, rate in zip(bars, conversion_rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                f'{rate:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Добавляем линии сравнения
    plt.axhline(conversion_rates[0], color='#e74c3c', linestyle='--', alpha=0.5, label='Базовый уровень')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 График A/B-теста сохранен: {save_path}")
    
    plt.show()

def calculate_roi(development_cost, monthly_maintenance, monthly_revenue_increase, months=12):
    """
    Расчет ROI (Return on Investment) рекомендательной системы
    
    Parameters:
    -----------
    development_cost : float
        Стоимость разработки
    monthly_maintenance : float
        Ежемесячные затраты на поддержку
    monthly_revenue_increase : float
        Ежемесячный прирост выручки
    months : int
        Период расчета в месяцах
        
    Returns:
    --------
    dict : Результаты расчета ROI
    """
    # Рассчитываем кумулятивные значения
    total_costs = development_cost + (monthly_maintenance * months)
    total_revenue_increase = monthly_revenue_increase * months
    
    # Чистая прибыль
    net_profit = total_revenue_increase - total_costs
    
    # ROI
    if total_costs > 0:
        roi = (net_profit / total_costs) * 100
    else:
        roi = float('inf')
    
    # Срок окупаемости (в месяцах)
    monthly_net = monthly_revenue_increase - monthly_maintenance
    if monthly_net > 0:
        payback_months = development_cost / monthly_net
    else:
        payback_months = float('inf')
    
    results = {
        'total_costs': total_costs,
        'total_revenue_increase': total_revenue_increase,
        'net_profit': net_profit,
        'roi_percent': roi,
        'payback_months': payback_months,
        'monthly_net': monthly_net
    }
    
    return results

if __name__ == "__main__":
    # Тестирование модуля
    print("Тестирование модуля ltv_calculator.py")
    print("=" * 50)
    
    # Расчет LTV для разных сценариев
    print("💰 Расчет LTV для разных сценариев:")
    ltv_comparison = calculate_ltv_scenarios()
    print(ltv_comparison.to_string(index=False))
    
    # Визуализация LTV
    plot_ltv_comparison(ltv_comparison)
    
    # A/B-тестирование
    print("\n🔬 Результаты A/B-тестирования:")
    ab_test_results = simulate_ab_test(n_users=20000)
    print(ab_test_results.to_string(index=False))
    
    # Визуализация A/B-теста
    plot_ab_test_results(ab_test_results)
    
    # Расчет ROI
    print("\n📈 Расчет ROI рекомендательной системы:")
    roi_results = calculate_roi(
        development_cost=500000,
        monthly_maintenance=50000,
        monthly_revenue_increase=1000000 * 0.17,  # +17% к месячной выручке
        months=12
    )
    
    print(f"   • Общие затраты за год: {roi_results['total_costs']:,.0f} руб.")
    print(f"   • Прирост выручки за год: {roi_results['total_revenue_increase']:,.0f} руб.")
    print(f"   • Чистая прибыль за год: {roi_results['net_profit']:,.0f} руб.")
    print(f"   • ROI за год: {roi_results['roi_percent']:.1f}%")
    print(f"   • Срок окупаемости: {roi_results['payback_months']:.1f} месяцев")