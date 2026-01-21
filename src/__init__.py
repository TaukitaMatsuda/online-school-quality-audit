"""
Рекомендательная система для онлайн-школы с анализом качества курсов
"""

__version__ = '1.0.0'
__author__ = 'Ekaterina Novikova'
__email__ = 'taukita.matsuda@gmail.com'

from .data_loader import load_purchase_data, connect_to_db
from .quality_metrics import create_synthetic_quality_metrics
from .recommender import CourseRecommender, analyze_joint_purchases
from .ltv_calculator import calculate_ltv_scenarios, simulate_ab_test

__all__ = [
    'load_purchase_data',
    'connect_to_db',
    'create_synthetic_quality_metrics',
    'CourseRecommender',
    'analyze_joint_purchases',
    'calculate_ltv_scenarios',
    'simulate_ab_test'
]