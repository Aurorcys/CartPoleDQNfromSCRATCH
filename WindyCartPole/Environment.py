import numpy as np 
import pandas as pd 

class CartPole:
    def __init__(self):
        self.gravity = 9.8
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.half_len = 0.5
        self.force_mag = 10.0
        self.tau = 0.02
        self.wind_force = 9.0
        self.reset()
    
    def reset(self):
        self.x = 0.0
        self.x_dot = 0.0
        self.theta = np.random.uniform(-.05, .05)
        self.theta_dot = 0.0
        return np.array([self.x, self.x_dot, self.theta, self.theta_dot])
    
    def step(self, action):
        force = self.force_mag if action == 1 else - self.force_mag

        total_mass = self.pole_mass + self.cart_mass
        pole_mass_len = self.half_len * self.pole_mass

        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        pole_wind_area = abs(cos_theta) * self.half_len * 2
        wind_force = pole_wind_area * self.wind_force

        total_force = force + wind_force

        temp = ((total_force + pole_mass_len * self.theta_dot ** 2 * sin_theta) / total_mass)

        theta_acc = (self.gravity * sin_theta - temp * cos_theta) / \
                    (self.half_len * (4.0/3.0 - self.pole_mass * cos_theta ** 2 / total_mass))
        
        x_acc = temp - pole_mass_len * cos_theta * theta_acc / total_mass

        self.x_dot += self.tau * x_acc
        self.theta_dot += self.tau * theta_acc

        self.x += self.x_dot * self.tau
        self.theta += self.theta_dot * self.tau

        done = abs(self.x) > 2.4 or abs(self.theta) > 0.3
        reward = 1.0 if not done else 0

        return np.array([self.x, self.x_dot, self.theta, self.theta_dot]), reward, done