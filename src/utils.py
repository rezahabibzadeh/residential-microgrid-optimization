"""
ماژول کمکی شامل توابع کاربردی برای پیش‌پردازش داده‌ها و محاسبات
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_and_preprocess_data(file_path):
    """
    بارگذاری و پیش‌پردازش داده‌های مصرف انرژی
    
    Parameters:
    file_path (str): مسیر فایل CSV
    
    Returns:
    pd.DataFrame: داده‌های پیش‌پردازش شده
    """
    # بارگذاری داده‌ها
    df = pd.read_csv(file_path)
    
    # تبدیل زمان به فرمت مناسب
    df['timestamp'] = pd.to_datetime(df['day'].astype(str) + ' ' + 
                                   (df['hour'] * 60).astype(int).astype(str) + 'min')
    
    # استخراج ویژگی‌های زمانی
    df['hour_of_day'] = df['hour']
    df['is_peak'] = ((df['hour'] >= 6) & (df['hour'] < 9)) | \
                   ((df['hour'] >= 17) & (df['hour'] < 22))
    
    # نرمال‌سازی مصرف انرژی
    scaler = MinMaxScaler()
    df['energy_normalized'] = scaler.fit_transform(df[['energy_consumption_kwh']])
    
    return df

def calculate_daily_consumption(df):
    """
    محاسبه مصرف روزانه هر خانوار
    
    Parameters:
    df (pd.DataFrame): داده‌های مصرف انرژی
    
    Returns:
    pd.DataFrame: مصرف روزانه هر خانوار
    """
    daily_consumption = df.groupby(['household_id', 'day'])['energy_consumption_kwh'].sum().reset_index()
    daily_stats = daily_consumption.groupby('household_id')['energy_consumption_kwh'].agg([
        'mean', 'std', 'min', 'max'
    ]).reset_index()
    
    return daily_stats

def create_consumption_features(df):
    """
    ایجاد ویژگی‌های جدید برای آنالیز مصرف
    
    Parameters:
    df (pd.DataFrame): داده‌های اصلی
    
    Returns:
    pd.DataFrame: داده‌های با ویژگی‌های جدید
    """
    # محاسبه مصرف متوسط ساعتی
    hourly_avg = df.groupby(['household_id', 'hour'])['energy_consumption_kwh'].mean().reset_index()
    hourly_avg = hourly_avg.rename(columns={'energy_consumption_kwh': 'hourly_avg_consumption'})
    
    # ادغام با داده‌های اصلی
    df = pd.merge(df, hourly_avg, on=['household_id', 'hour'])
    
    # محاسبه انحراف از میانگین
    df['deviation_from_avg'] = df['energy_consumption_kwh'] - df['hourly_avg_consumption']
    
    return df

def calculate_cost_savings(original_consumption, optimized_consumption, tariff_rates):
    """
    محاسبه صرفه‌جویی هزینه پس از بهینه‌سازی
    
    Parameters:
    original_consumption (array): مصرف اولیه
    optimized_consumption (array): مصرف بهینه‌شده
    tariff_rates (array): نرخ تعرفه برق
    
    Returns:
    dict: نتایج محاسبات صرفه‌جویی
    """
    original_cost = np.sum(original_consumption * tariff_rates)
    optimized_cost = np.sum(optimized_consumption * tariff_rates)
    
    savings = original_cost - optimized_cost
    savings_percentage = (savings / original_cost) * 100
    
    return {
        'original_cost': original_cost,
        'optimized_cost': optimized_cost,
        'savings_amount': savings,
        'savings_percentage': savings_percentage
    }
