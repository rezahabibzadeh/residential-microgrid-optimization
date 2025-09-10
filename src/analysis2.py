"""
ماژول آنالیز داده‌ها و محاسبات آماری
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def analyze_consumption_patterns(df):
    """
    آنالیز الگوهای مصرف انرژی
    
    Parameters:
    df (pd.DataFrame): داده‌های مصرف انرژی
    
    Returns:
    dict: نتایج آنالیز
    """
    results = {}
    
    # آماره‌های توصیفی
    results['descriptive_stats'] = df['energy_consumption_kwh'].describe().to_dict()
    
    # مصرف کل
    results['total_consumption'] = df['energy_consumption_kwh'].sum()
    
    # مصرف متوسط روزانه
    daily_consumption = df.groupby('day')['energy_consumption_kwh'].sum()
    results['avg_daily_consumption'] = daily_consumption.mean()
    
    # پیک مصرف
    results['peak_consumption'] = df['energy_consumption_kwh'].max()
    
    # آنالیز ساعتی
    hourly_stats = df.groupby('hour')['energy_consumption_kwh'].agg(['mean', 'std', 'max'])
    results['hourly_analysis'] = hourly_stats.to_dict()
    
    return results

def cluster_households(df, n_clusters=3):
    """
    خوشه‌بندی خانوارها بر اساس الگوی مصرف
    
    Parameters:
    df (pd.DataFrame): داده‌های مصرف انرژی
    n_clusters (int): تعداد خوشه‌ها
    
    Returns:
    pd.DataFrame: داده‌های با برچسب خوشه
    """
    # ایجاد ویژگی‌های برای خوشه‌بندی
    features = df.groupby('household_id').agg({
        'energy_consumption_kwh': ['mean', 'std', 'max'],
        'appliance_1_status': 'mean',
        'appliance_2_status': 'mean',
        'appliance_3_status': 'mean'
    }).reset_index()
    
    features.columns = ['household_id', 'avg_consumption', 'std_consumption', 
                       'max_consumption', 'app1_usage', 'app2_usage', 'app3_usage']
    
    # نرمال‌سازی ویژگی‌ها
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features.drop('household_id', axis=1))
    
    # خوشه‌بندی
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    features['cluster'] = kmeans.fit_predict(scaled_features)
    
    return features

def calculate_energy_statistics(df):
    """
    محاسبه آماره‌های انرژی
    
    Parameters:
    df (pd.DataFrame): داده‌های مصرف انرژی
    
    Returns:
    dict: آماره‌های محاسبه شده
    """
    stats = {}
    
    # محاسبه بر اساس خانوار
    household_stats = df.groupby('household_id')['energy_consumption_kwh'].agg([
        'mean', 'std', 'min', 'max', 'sum'
    ]).rename(columns={
        'mean': 'avg_consumption',
        'std': 'consumption_std',
        'min': 'min_consumption',
        'max': 'max_consumption',
        'sum': 'total_consumption'
    })
    
    stats['household_stats'] = household_stats
    
    # محاسبه بر اساس ساعت
    hourly_stats = df.groupby('hour')['energy_consumption_kwh'].agg([
        'mean', 'std', 'sum'
    ])
    stats['hourly_stats'] = hourly_stats
    
    # محاسبه نرمال‌شده
    df['normalized_consumption'] = (df['energy_consumption_kwh'] - 
                                  df['energy_consumption_kwh'].mean()) / df['energy_consumption_kwh'].std()
    
    return stats

def perform_statistical_tests(df):
    """
    انجام آزمون‌های آماری
    
    Parameters:
    df (pd.DataFrame): داده‌های مصرف انرژی
    
    Returns:
    dict: نتایج آزمون‌ها
    """
    results = {}
    
    # آزمون نرمالیتی
    stat, p_value = stats.shapiro(df['energy_consumption_kwh'].sample(1000))
    results['normality_test'] = {'statistic': stat, 'p_value': p_value}
    
    # مقایسه مصرف ساعات پیک و غیرپیک
    peak_consumption = df[df['is_peak']]['energy_consumption_kwh']
    off_peak_consumption = df[~df['is_peak']]['energy_consumption_kwh']
    
    t_stat, p_value = stats.ttest_ind(peak_consumption, off_peak_consumption)
    results['peak_vs_off_peak'] = {'t_statistic': t_stat, 'p_value': p_value}
    
    return results

def generate_optimization_report(original_df, optimized_df):
    """
    تولید گزارش بهینه‌سازی
    
    Parameters:
    original_df (pd.DataFrame): داده‌های اولیه
    optimized_df (pd.DataFrame): داده‌های بهینه‌شده
    
    Returns:
    dict: گزارش بهینه‌سازی
    """
    report = {}
    
    # محاسبه کاهش مصرف
    original_total = original_df['energy_consumption_kwh'].sum()
    optimized_total = optimized_df['energy_consumption_kwh'].sum()
    reduction = original_total - optimized_total
    reduction_percentage = (reduction / original_total) * 100
    
    report['consumption_reduction'] = {
        'original_total': original_total,
        'optimized_total': optimized_total,
        'reduction_kwh': reduction,
        'reduction_percentage': reduction_percentage
    }
    
    # محاسبه کاهش پیک مصرف
    original_peak = original_df['energy_consumption_kwh'].max()
    optimized_peak = optimized_df['energy_consumption_kwh'].max()
    peak_reduction = original_peak - optimized_peak
    
    report['peak_reduction'] = {
        'original_peak': original_peak,
        'optimized_peak': optimized_peak,
        'peak_reduction_kwh': peak_reduction
    }
    
    # ذخیره گزارش
    report_df = pd.DataFrame.from_dict(report, orient='index')
    report_df.to_csv('../results/optimization_report.csv')
    
    return report
