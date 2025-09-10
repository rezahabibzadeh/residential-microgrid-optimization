"""
اسکریپت اصلی برای اجرای آنالیز و شبیه‌سازی
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ipso_algorithm import ImprovedPSO

def main():
    print("Starting Energy Management Optimization...")
    
    # تعریف تابع هدف (می‌توانید با تابع واقعی جایگزین کنید)
    def objective_function(x):
        return np.sum(x**2) + (x[0] - 1)**2 + (x[1] + 0.5)**2
    
    # تعریف محدودیت‌ها
    bounds = [(-2.0, 2.0), (-1.5, 1.5), (-1.0, 1.0)]
    
    # ایجاد و اجرای الگوریتم IPSO
    ipso = ImprovedPSO(
        n_particles=30,
        max_iter=100,
        bounds=bounds,
        objective_func=objective_function
    )
    
    best_solution, best_score = ipso.optimize()
    
    # نمایش نتایج
    print(f"Best Solution: {best_solution}")
    print(f"Best Score: {best_score}")
    
    # رسم نمودار همگرایی
    plt.figure(figsize=(10, 6))
    plt.plot(ipso.convergence_history)
    plt.title('IPSO Convergence History')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness Value')
    plt.grid(True)
    plt.savefig('../results/convergence_plot.png')
    plt.show()

if __name__ == "__main__":
    main()
