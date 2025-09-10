"""
پیاده‌سازی الگوریتم بهبودیافته ازدحام ذرات (IPSO)
برای بهینه‌سازی مصرف انرژی در ریزشبکه‌های مسکونی
"""

import numpy as np

class ImprovedPSO:
    def __init__(self, n_particles, max_iter, bounds, objective_func):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.bounds = np.array(bounds)
        self.objective_func = objective_func
        self.dimension = len(bounds)
        
        # Initialize parameters
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1_max = 2.5
        self.c1_min = 0.5
        self.c2_max = 2.5
        self.c2_min = 0.5
        
        # Initialize swarm
        self.initialize_swarm()
        
    def initialize_swarm(self):
        """مقداردهی اولیه ذرات"""
        self.positions = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            (self.n_particles, self.dimension)
        )
        self.velocities = np.zeros((self.n_particles, self.dimension))
        
        # Initialize personal best
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.array([self.objective_func(p) for p in self.positions])
        
        # Initialize global best
        self.gbest_index = np.argmin(self.pbest_scores)
        self.gbest_position = self.pbest_positions[self.gbest_index]
        self.gbest_score = self.pbest_scores[self.gbest_index]
        
        self.convergence_history = []
    
    def optimize(self):
        """اجرای الگوریتم بهینه‌سازی"""
        for iteration in range(self.max_iter):
            # Update inertia weight
            w = self.w_max - (self.w_max - self.w_min) * (iteration / self.max_iter)
            
            for i in range(self.n_particles):
                # Update coefficients
                c1 = self.c1_min + (self.c1_max - self.c1_min) * np.random.rand()
                c2 = self.c2_min + (self.c2_max - self.c2_min) * np.random.rand()
                
                # Update velocity
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive = c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social = c2 * r2 * (self.gbest_position - self.positions[i])
                self.velocities[i] = w * self.velocities[i] + cognitive + social
                
                # Update position
                self.positions[i] += self.velocities[i]
                
                # Apply bounds
                self.positions[i] = np.clip(self.positions[i], 
                                          self.bounds[:, 0], 
                                          self.bounds[:, 1])
                
                # Evaluate fitness
                current_score = self.objective_func(self.positions[i])
                
                # Update personal best
                if current_score < self.pbest_scores[i]:
                    self.pbest_positions[i] = self.positions[i]
                    self.pbest_scores[i] = current_score
                    
                    # Update global best
                    if current_score < self.gbest_score:
                        self.gbest_position = self.positions[i]
                        self.gbest_score = current_score
            
            self.convergence_history.append(self.gbest_score)
            
        return self.gbest_position, self.gbest_score

# تابع هدف نمونه برای تست
def sample_objective_function(x):
    """تابع هدف نمونه برای بهینه‌سازی"""
    return np.sum(x**2) + np.random.rand() * 0.1
