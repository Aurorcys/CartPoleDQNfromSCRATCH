import numpy as np
import gymnasium as gym
from NeuralNetwork import CartPoleDQN

env = gym.make('CartPole-v1')
model = CartPoleDQN()
target_model = CartPoleDQN()  # Target network

# Sync target
target_model.W1, target_model.b1 = model.W1.copy(), model.b1.copy()
target_model.W2, target_model.b2 = model.W2.copy(), model.b2.copy()
target_model.W3, target_model.b3 = model.W3.copy(), model.b3.copy()

epsilon = 1.0
gamma = 0.95
episodes = 500
replay_buffer = []
max_buffer = 2000
batch_size = 32
scores = []
step_count = 0

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for t in range(500):
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = model.forward(state)
            action = np.argmax(q_values)

        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        replay_buffer.append((state, action, reward, next_state, done))
        if len(replay_buffer) > max_buffer:
            replay_buffer.pop(0)

        if len(replay_buffer) >= batch_size:
            for _ in range(4):
                batch = np.random.choice(len(replay_buffer), batch_size, replace=False)
                for idx in batch:
                    s, a, r, ns, d = replay_buffer[idx]
                    
                    if d:
                        target = r
                    else:
                        target = r + gamma * np.max(target_model.forward(ns))
                    
                    # Clip target to reasonable range
                    target = np.clip(target, -100, 100)
                    
                    model.forward(s)
                    model.backward(target, a)
                    step_count += 1
                    
                    # Sync target network
                    if step_count % 250 == 0:
                        target_model.W1 = model.W1.copy()
                        target_model.b1 = model.b1.copy()
                        target_model.W2 = model.W2.copy()
                        target_model.b2 = model.b2.copy()
                        target_model.W3 = model.W3.copy()
                        target_model.b3 = model.b3.copy()

        state = next_state
        if done:
            break

    scores.append(total_reward)
    epsilon = max(0.01, epsilon * 0.997)

    if episode % 50 == 0:
        print(f"Episode {episode}, Avg: {np.mean(scores[-50:]):.1f}, Epsilon: {epsilon:.3f}")

print("Training complete!")
env.close()