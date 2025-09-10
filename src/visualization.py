"""
ماژول visualization برای رسم نمودارها و گراف‌ها
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_consumption_patterns(df, household_id=1):
    """
    رسم الگوی مصرف انرژی برای یک خانوار خاص
    
    Parameters:
    df (pd.DataFrame): داده‌های مصرف انرژی
    household_id (int): شماره خانوار
    """
    household_data = df[df['household_id'] == household_id]
    
    plt.figure(figsize=(15, 8))
    
    # مصرف ساعتی متوسط
    hourly_avg = household_data.groupby('hour')['energy_consumption_kwh'].mean()
    
    plt.subplot(2, 2, 1)
    hourly_avg.plot(kind='bar', color='skyblue')
    plt.title(f'میانگین مصرف ساعتی - خانوار {household_id}')
    plt.xlabel('ساعت روز')
    plt.ylabel('مصرف انرژی (kWh)')
    plt.xticks(rotation=45)
    
    # توزیع مصرف
    plt.subplot(2, 2, 2)
    plt.hist(household_data['energy_consumption_kwh'], bins=30, alpha=0.7, color='lightgreen')
    plt.title('توزیع مصرف انرژی')
    plt.xlabel('مصرف انرژی (kWh)')
    plt.ylabel('تعداد')
    
    # مصرف روزانه
    daily_consumption = household_data.groupby('day')['energy_consumption_kwh'].sum()
    plt.subplot(2, 2, 3)
    daily_consumption.plot(kind='line', color='orange')
    plt.title('مصرف روزانه')
    plt.xlabel('روز')
    plt.ylabel('مصرف کل (kWh)')
    
    # وضعیت دستگاه‌ها
    appliance_cols = ['appliance_1_status', 'appliance_2_status', 'appliance_3_status']
    appliance_usage = household_data[appliance_cols].mean()
    plt.subplot(2, 2, 4)
    appliance_usage.plot(kind='bar', color=['red', 'green', 'blue'])
    plt.title('میانگین استفاده از دستگاه‌ها')
    plt.xlabel('دستگاه')
    plt.ylabel('نسبت استفاده')
    
    plt.tight_layout()
    plt.savefig(f'../results/household_{household_id}_consumption_patterns.png')
    plt.show()

def plot_convergence(convergence_history, algorithm_name='IPSO'):
    """
    رسم نمودار همگرایی الگوریتم
    
    Parameters:
    convergence_history (list): تاریخچه مقادیر تابع هدف
    algorithm_name (str): نام الگوریتم
    """
    plt.figure(figsize=(10, 6))
    plt.plot(convergence_history, linewidth=2)
    plt.title(f'نمودار همگرایی الگوریتم {algorithm_name}')
    plt.xlabel('تعداد تکرار')
    plt.ylabel('مقدار تابع هدف')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'../results/{algorithm_name}_convergence.png')
    plt.show()

def plot_comparison_results(results_dict):
    """
    رسم نمودار مقایسه‌ای بین سناریوهای مختلف
    
    Parameters:
    results_dict (dict): دیکشنری شامل نتایج سناریوها
    """
    scenarios = list(results_dict.keys())
    costs = [results_dict[scenario]['total_cost'] for scenario in scenarios]
    peak_demands = [results_dict[scenario]['peak_demand'] for scenario in scenarios]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # نمودار هزینه
    ax1.bar(scenarios, costs, color='lightblue')
    ax1.set_title('مقایسه هزینه کل سناریوها')
    ax1.set_ylabel('هزینه (واحد پولی)')
    ax1.tick_params(axis='x', rotation=45)
    
    # نمودار پیک مصرف
    ax2.bar(scenarios, peak_demands, color='lightcoral')
    ax2.set_title('مقایسه پیک مصرف سناریوها')
    ax2.set_ylabel('پیک مصرف (kWh)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('../results/scenario_comparison.png')
    plt.show()

def create_summary_report(df, results):
    """
    ایجاد گزارش خلاصه آنالیز
    
    Parameters:
    df (pd.DataFrame): داده‌های مصرف انرژی
    results (dict): نتایج بهینه‌سازی
    """
    report = {
        'total_households': df['household_id'].nunique(),
        'total_days': df['day'].nunique(),
        'total_records': len(df),
        'average_daily_consumption': df.groupby('day')['energy_consumption_kwh'].sum().mean(),
        'peak_consumption': df['energy_consumption_kwh'].max(),
        'optimization_savings': results.get('savings_percentage', 0)
    }
    
    # ذخیره گزارش
    report_df = pd.DataFrame.from_dict(report, orient='index', columns=['Value'])
    report_df.to_csv('../results/summary_report.csv')
    
    return report
