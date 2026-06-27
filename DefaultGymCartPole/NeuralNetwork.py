import numpy as np
import pandas as pd 

class CartPoleDQN:
    def __init__(self):
        self.W1 = np.random.randn(4, 24) * np.sqrt(2.0 / 4)
        self.b1 = np.zeros(24)
        
        self.W2 = np.random.randn(24, 24) * np.sqrt(2.0 / 24)
        self.b2 = np.zeros(24)
        
        self.W3 = np.random.randn(24, 2) * np.sqrt(2.0 / 24)
        self.b3 = np.zeros(2)
    def forward(self, state):
        self.state = state

        self.z1 = state @ self.W1 + self.b1
        self.r1 = np.maximum(self.z1, 0)
        self.z2 = self.r1 @ self.W2 + self.b2
        self.r2 = np.maximum(self.z2, 0)
        q_values = self.r2 @ se lf.W3 + self.b3

        return q_values
    

    def backward(self, target, chosen_idx):
        q_chosen = self.r2 @ self.W3[:, chosen_idx] + self.b3[chosen_idx]
        d_loss = -2 * (target - q_chosen)

        d_z3 = np.zeros(2)
        d_z3[chosen_idx] = d_loss

        #layer three
        d_W3 = np.outer(self.r2, d_z3)
        d_b3 = d_z3
        d_r2 = d_z3 @ self.W3.T

        #layer two
        d_z2 = d_r2 * (self.z2 > 0)
        d_W2 = np.outer(self.r1, d_z2)
        d_b2 = d_z2
        d_r1 = d_z2 @ self.W2.T 

        #layer one
        d_z1 = d_r1 * (self.z1 > 0)
        d_W1 = np.outer(self.state, d_z1)
        d_b1 = d_z1 

        lr = 0.01
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W3 -= lr * d_W3
        self.b3 -= lr * d_b3