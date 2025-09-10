"""
ماژول تولید داده‌های مصنوعی مصرف انرژی
بر اساس الگوهای واقعی مصرف شبیه‌سازی شده است
"""

import numpy as np
import pandas as pd

class EnergyDataGenerator:
    def __init__(self, n_households=50, n_days=90):
        self.n_households = n_households
        self.n_days = n_days
        self.n_intervals = 24 * 4  # 15-minute intervals
        
    def generate_consumption_pattern(self, hour):
        """تولید الگوی مصرف بر اساس ساعت روز"""
        if 6 <= hour < 9:  # پیک صبح
            return np.random.uniform(1.2, 1.8)
        elif 17 <= hour < 22:  # پیک عصر
            return np.random.uniform(1.5, 2.2)
        else:  # مصرف عادی
            return np.random.uniform(0.3, 0.8)
    
    def generate_dataset(self):
        """تولید دیتاست کامل"""
        data = []
        for household_id in range(1, self.n_households + 1):
            base_load = np.random.uniform(0.5, 2.0)
            
            for day in range(1, self.n_days + 1):
                for interval in range(self.n_intervals):
                    hour = interval / 4
                    load_factor = self.generate_consumption_pattern(hour)
                    noise = np.random.normal(0, 0.1)
                    total_load = base_load * load_factor + noise
                    
                    data.append({
                        'household_id': household_id,
                        'day': day,
                        'time_interval': interval,
                        'hour': hour,
                        'energy_consumption_kwh': max(0, total_load),
                        'appliance_1_status': np.random.choice([0, 1]),
                        'appliance_2_status': np.random.choice([0, 1]),
                        'appliance_3_status': np.random.choice([0, 1])
                    })
        
        return pd.DataFrame(data)

# تولید و ذخیره‌سازی داده‌ها
if __name__ == "__main__":
    generator = EnergyDataGenerator()
    df = generator.generate_dataset()
    df.to_csv('../data/household_energy_data.csv', index=False)
    print("Dataset generated successfully!")
