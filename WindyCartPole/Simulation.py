import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import gymnasium as gym

class DQN:
    def __init__(self):
        scale1 = np.sqrt(2.0 / 4)
        self.W1 = np.random.randn(4, 64) * scale1
        self.b1 = np.zeros((1, 64))
        
        scale2 = np.sqrt(2.0 / 64)
        self.W2 = np.random.randn(64, 64) * scale2
        self.b2 = np.zeros((1, 64))
        
        scale3 = np.sqrt(2.0 / 64)
        self.W3 = np.random.randn(64, 2) * scale3
        self.b3 = np.zeros((1, 2))
    
    def forward(self, s):
        if s.ndim == 1:
            s = s.reshape(1, -1)
        self.s = s
        self.z1 = s @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)  # Tanh instead of ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.tanh(self.z2)  # Tanh instead of ReLU
        return self.a2 @ self.W3 + self.b3
    
    def train(self, s, a, target, lr=0.001):
        q = self.forward(s)
        q_target = q.copy()
        
        batch_indices = np.arange(len(a))
        q_target[batch_indices, a] = target
        
        d_q = 2 * (q - q_target) / len(a)
        
        d_W3 = self.a2.T @ d_q
        d_b3 = np.sum(d_q, axis=0, keepdims=True)
        
        d_a2 = d_q @ self.W3.T
        d_z2 = d_a2 * (1 - self.a2 ** 2)  # Tanh derivative
        
        d_W2 = self.a1.T @ d_z2
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)
        
        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * (1 - self.a1 ** 2)  # Tanh derivative
        
        d_W1 = self.s.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)
        
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W3 -= lr * d_W3
        self.b3 -= lr * d_b3

from Environment import CartPole
env = CartPole()
model = DQN()
target = DQN()

def sync_weights(src, dst):
    dst.W1, dst.b1 = src.W1.copy(), src.b1.copy()
    dst.W2, dst.b2 = src.W2.copy(), src.b2.copy()
    dst.W3, dst.b3 = src.W3.copy(), src.b3.copy()

sync_weights(model, target)

buffer = []
max_buffer = 20000
batch_size = 64
epsilon = 1.0
gamma = 0.99
scores = []
step = 0

for ep in range(3000):
    s = env.reset()
    score = 0
    
    for _ in range(500):
        if np.random.random() < epsilon:
            a = np.random.randint(2)
        else:
            a = np.argmax(model.forward(s))
        
        ns, r, done = env.step(a)
        score += r
        
        buffer.append((s, a, r, ns, done))
        if len(buffer) > max_buffer:
            buffer.pop(0)
            
        if len(buffer) >= batch_size:
            idxs = np.random.choice(len(buffer), batch_size, replace=False)
            
            # FIXED: Correctly pulling indices from the tuples
            batch_s = np.array([buffer[i][0] for i in idxs])
            batch_a = np.array([buffer[i][1] for i in idxs], dtype=int)
            batch_r = np.array([buffer[i][2] for i in idxs])
            batch_ns = np.array([buffer[i][3] for i in idxs])
            batch_d = np.array([buffer[i][4] for i in idxs], dtype=bool)
            
            target_q_next = target.forward(batch_ns)
            max_target_q_next = np.max(target_q_next, axis=1)
            
            targets = batch_r + gamma * max_target_q_next * (~batch_d)
            
            model.train(batch_s, batch_a, targets, lr=0.001)
            step += 1
            
            if step % 100 == 0:
                sync_weights(model, target)
        
        s = ns
        if done:
            break
            
    scores.append(score)
    epsilon = max(0.01, epsilon * 0.995)
    
    if ep % 50 == 0:
        avg = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
        print(f"Ep {ep}, Avg: {avg:.1f}, Eps: {epsilon:.3f}")





# ============================================================
# TRAINING PROGRESS PLOT
# ============================================================
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#0d0d1a')
ax.set_facecolor('#0d0d1a')

ax.plot(scores, color='#00d4ff', alpha=0.3, linewidth=0.5)
window = 50
rolling = np.convolve(scores, np.ones(window)/window, mode='valid')
ax.plot(range(window-1, len(scores)), rolling, color='#ff6b6b', linewidth=2.5, label=f'{window}-Episode Avg')

ax.axhline(195, color='#00ff88', linestyle='--', linewidth=1.5, label='Solved (195)')
ax.axhline(500, color='#ffcc00', linestyle='--', linewidth=1.5, label='Max (500)')
ax.set_xlabel('Episode', fontsize=14, color='white')
ax.set_ylabel('Score', fontsize=14, color='white')
ax.set_title('DQN CartPole — From Zero to Perfect Balance', fontsize=18, color='white', fontweight='bold')
ax.legend(facecolor='#1a1a3e', edgecolor='white', fontsize=12)
ax.grid(True, alpha=0.1, color='white')
ax.tick_params(colors='white')

plt.tight_layout()
plt.savefig('cartpole_training.png', dpi=300, bbox_inches='tight', facecolor='#0d0d1a')
plt.show()






from Environment import CartPole
env = CartPole()
s = env.reset()

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#0d0d1a')
ax.set_facecolor('#0d0d1a')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-0.5, 2.0)
ax.axis('off')

# Track
track = ax.plot([-2.4, 2.4], [0, 0], 'white', linewidth=3)[0]

# Cart
cart_width, cart_height = 0.4, 0.2
cart = plt.Rectangle((-cart_width/2, -cart_height/2), cart_width, cart_height, 
                      fill=True, facecolor='#00d4ff', edgecolor='white', linewidth=2)
ax.add_patch(cart)

# Pole
pole, = ax.plot([], [], 'r-', linewidth=4)

# Wind particles
wind_particles = []
for _ in range(20):
    x = np.random.uniform(2.0, 2.5)
    y = np.random.uniform(0.0, 2.0)
    particle, = ax.plot([x], [y], 'o', color='white', alpha=0.3, markersize=3)
    wind_particles.append({'obj': particle, 'x': x, 'y': y, 'speed': np.random.uniform(0.02, 0.08)})

# Wind arrows
wind_arrows = []
for i in range(5):
    arrow_y = i * 0.4 + 0.2
    arrow, = ax.plot([], [], 'w-', linewidth=1.5, alpha=0.4)
    wind_arrows.append({'obj': arrow, 'y': arrow_y})

# Wind strength bar
wind_bar_bg = plt.Rectangle((1.5, 1.75), 0.8, 0.15, fill=True, facecolor='#333344', edgecolor='white', linewidth=1)
ax.add_patch(wind_bar_bg)
wind_bar_fill = plt.Rectangle((1.5, 1.75), 0.0, 0.15, fill=True, facecolor='#00d4ff')
ax.add_patch(wind_bar_fill)
wind_label = ax.text(1.5, 1.95, '', fontsize=10, color='white', fontfamily='monospace')

# Stats
score_text = ax.text(-2.3, 1.8, '', fontsize=16, color='white', fontfamily='monospace')
step_text = ax.text(0.3, 1.8, '', fontsize=16, color='white', fontfamily='monospace')

def animate(frame):
    global s
    a = np.argmax(model.forward(s))
    s, _, done = env.step(a)
    
    x = s[0]
    theta = s[2]
    wind = env.wind_force * abs(np.cos(theta)) * env.half_len * 2
    
    # Update cart and pole
    cart.set_x(x - cart_width/2)
    pole_x = [x, x + np.sin(theta) * 1.0]
    pole_y = [0, np.cos(theta) * 1.0]
    pole.set_data(pole_x, pole_y)
    
    # Animate wind particles (blow LEFT to RIGHT)
    for p in wind_particles:
        p['x'] += p['speed']  # Move rightward
        if p['x'] > 2.5:
            p['x'] = -2.5
            p['y'] = np.random.uniform(0.0, 2.0)
            p['speed'] = np.random.uniform(0.03, 0.1)
        p['obj'].set_data([p['x']], [p['y']])
        p['obj'].set_alpha(0.1 + min(wind / 15.0, 0.8))

    # Wind arrows pointing RIGHT
    for i, arrow in enumerate(wind_arrows):
        arrow_len = 0.3 + wind * 0.02
        arrow_y = arrow['y']
        arrow_x_start = -1.0 + (frame * 0.02) % 3.0  # Start from left
        arrow['obj'].set_data([arrow_x_start - arrow_len, arrow_x_start], [arrow_y, arrow_y])
        arrow['obj'].set_alpha(0.1 + min(wind / 15.0, 0.6))
        
    # Update wind strength bar
    wind_pct = min(wind / 15.0, 1.0)
    wind_bar_fill.set_width(0.8 * wind_pct)
    wind_label.set_text(f'Wind: {wind:.1f}N')
    
    score_text.set_text(f'Score: {frame}')
    step_text.set_text(f'Angle: {theta:.2f}')
    
    if done:
        s = env.reset()
    
    return cart, pole, score_text, step_text, wind_label

ani = animation.FuncAnimation(fig, animate, frames=500, interval=20, blit=False)
plt.title('DQN CartPole — Windy Balance', fontsize=18, color='white', fontweight='bold', pad=20)
plt.show()
