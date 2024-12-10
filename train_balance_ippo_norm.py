import vmas
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import pygame
import time
from collections import deque
import matplotlib.pyplot as plt

# 观察值标准化器
class ObservationNormalizer:
    def __init__(self, shape, device):
        self.device = device
        self.running_mean = torch.zeros(shape).to(device)
        self.running_var = torch.ones(shape).to(device)
        self.count = 1e-4
        
    def update(self, obs):
        batch_mean = obs.mean(0)
        batch_var = obs.var(0)
        batch_count = obs.shape[0]
        
        delta = batch_mean - self.running_mean
        self.running_mean += delta * batch_count / (self.count + batch_count)
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / (self.count + batch_count)
        self.running_var = M2 / (self.count + batch_count)
        self.count += batch_count
        
    def normalize(self, obs):
        return (obs - self.running_mean) / (torch.sqrt(self.running_var) + 1e-8)

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(),
        )
        self.mean_layer = nn.Linear(hidden_dim // 2, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim) - 1.0)
        
        # 使用正交初始化
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0)
    
    def forward(self, obs):
        features = self.net(obs)
        mean = torch.tanh(self.mean_layer(features))
        std = torch.exp(self.log_std)
        return mean, std

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, obs):
        return self.net(obs)

class IPPOAgent:
    def __init__(self, obs_dim, action_dim, agent_id, device="cpu"):
        self.device = device
        self.agent_id = agent_id
        
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(obs_dim).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # PPO参数
        self.clip_param = 0.2
        self.ppo_epochs = 10
        self.num_mini_batches = 4
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
        # 添加观察值标准化器
        self.obs_normalizer = ObservationNormalizer(obs_dim, device)
        
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def select_action(self, obs):
        with torch.no_grad():
            # 使用标准化后的观察值
            normalized_obs = self.obs_normalizer.normalize(obs)
            mean, std = self.actor(normalized_obs)
            dist = Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            value = self.critic(normalized_obs)
            
        return action, log_prob, value
    
    def store_transition(self, obs, action, reward, value, log_prob, done):
        # 更新观察值统计信息
        self.obs_normalizer.update(obs)
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward.reshape(-1, 1))
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done.reshape(-1, 1).float())
    
    def clear_memory(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def compute_gae(self):
        T = len(self.rewards)
        num_envs = self.rewards[0].shape[0]
        advantages = torch.zeros(T, num_envs, 1, device=self.device)
        returns = torch.zeros(T, num_envs, 1, device=self.device)
        last_gae_lam = torch.zeros(num_envs, 1, device=self.device)
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = torch.zeros_like(self.values[0])
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * last_gae_lam
            returns[t] = advantages[t] + self.values[t]
        
        advantages = advantages.reshape(-1, 1)
        returns = returns.reshape(-1, 1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self):
        advantages, returns = self.compute_gae()
        observations = torch.cat(self.observations)
        # 使用标准化后的观察值进行更新
        normalized_obs = self.obs_normalizer.normalize(observations)
        actions = torch.cat(self.actions)
        old_log_probs = torch.cat(self.log_probs)
        
        total_actor_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        batch_size = len(self.observations) * self.rewards[0].shape[0]
        mini_batch_size = batch_size // self.num_mini_batches
        
        for _ in range(self.ppo_epochs):
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_indices = indices[start:end]
                
                mb_obs = normalized_obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                mean, std = self.actor(mb_obs)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(mb_actions).sum(-1, keepdim=True)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                value_pred = self.critic(mb_obs)
                value_loss = 0.5 * ((mb_returns - value_pred) ** 2).mean()
                
                loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        self.clear_memory()
        
        num_updates = self.ppo_epochs * (batch_size // mini_batch_size)
        return total_actor_loss / num_updates, total_value_loss / num_updates, total_entropy / num_updates

class IPPOTrainer:
    def __init__(self, env, device="cpu"):
        self.env = env
        self.device = device
        self.n_agents = len(env.agents)
        
        obs = env.reset()
        self.obs_dim = obs[0].shape[1]
        self.action_dim = env.agents[0].action_size
        
        print(f"观察空间维度: {self.obs_dim}")
        print(f"动作空间维度: {self.action_dim}")
        print(f"智能体数量: {self.n_agents}")
        
        self.agents = [IPPOAgent(self.obs_dim, self.action_dim, i, device) for i in range(self.n_agents)]
    
    def train_episode(self):
        obs = self.env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            actions = []
            for i, agent in enumerate(self.agents):
                action, log_prob, value = agent.select_action(obs[i])
                actions.append(action)
                
                agent.store_transition(
                    obs[i],
                    action,
                    rewards[i] if 'rewards' in locals() else torch.zeros(self.env.num_envs, device=self.device),
                    value,
                    log_prob,
                    torch.zeros(self.env.num_envs, device=self.device)
                )
            
            next_obs, rewards, dones, _ = self.env.step(actions)
            done = any(d.any() for d in dones)
            
            for i in range(self.n_agents):
                agent.rewards[-1] = rewards[i].reshape(-1, 1)
                agent.dones[-1] = torch.tensor(done, device=self.device, dtype=torch.float).expand(self.env.num_envs).reshape(-1, 1)
                episode_reward += rewards[i].mean().item()
            
            obs = next_obs
        
        actor_losses = []
        value_losses = []
        entropies = []
        
        for agent in self.agents:
            actor_loss, value_loss, entropy = agent.update()
            actor_losses.append(actor_loss)
            value_losses.append(value_loss)
            entropies.append(entropy)
        
        return np.mean(actor_losses), np.mean(value_losses), np.mean(entropies), episode_reward

def plot_rewards(rewards, window_size=100):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.4, label='Raw Rewards')
    plt.plot(np.convolve(rewards, np.ones(window_size)/window_size, mode='valid'), 
             label=f'Average over {window_size} episodes')
    plt.title('IPPO with Observation Normalization Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('ippo_norm_training_rewards.png')
    plt.close()

def main():
    env = vmas.make_env(
        scenario="balance",
        num_envs=16,
        device="cpu",
        continuous_actions=True,
        seed=0,
        n_agents=4
    )
    
    trainer = IPPOTrainer(env)
    num_episodes = 500
    
    pygame.init()
    frame = env.render(mode="rgb_array")
    height, width = frame.shape[:2]
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("VMAS Balance IPPO Training (with Observation Normalization)")
    clock = pygame.time.Clock()
    
    all_rewards = []
    episode_rewards = deque(maxlen=100)
    best_avg_reward = float('-inf')
    
    log_file = open('ippo_norm_training_log.txt', 'w')
    render_interval = 1
    frame_rate = 60
    
    for episode in range(num_episodes):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                log_file.close()
                return
        
        actor_loss, value_loss, entropy, episode_reward = trainer.train_episode()
        
        all_rewards.append(episode_reward)
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards)
        
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            checkpoint = {
                'episode': episode,
                'agents_state': [
                    {
                        'actor_state_dict': agent.actor.state_dict(),
                        'critic_state_dict': agent.critic.state_dict(),
                        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                        'obs_normalizer_state': {
                            'running_mean': agent.obs_normalizer.running_mean,
                            'running_var': agent.obs_normalizer.running_var,
                            'count': agent.obs_normalizer.count
                        }
                    }
                    for agent in trainer.agents
                ],
                'best_avg_reward': best_avg_reward,
            }
            torch.save(checkpoint, 'ippo_norm_best_model.pt')
        
        log_info = f"Episode {episode + 1}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}\n"
        log_info += f"Actor Loss: {actor_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy: {entropy:.4f}\n"
        log_info += f"Best Avg Reward: {best_avg_reward:.2f}\n"
        
        print(log_info)
        log_file.write(log_info + '\n')
        log_file.flush()
        
        if episode % render_interval == 0:
            try:
                frame = env.render(mode="rgb_array")
                surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                screen.blit(surface, (0, 0))
                pygame.display.flip()
                clock.tick(frame_rate)
            except Exception as e:
                print(f"渲染错误: {e}")
                log_file.write(f"渲染错误: {e}\n")
        
        if (episode + 1) % 100 == 0:
            plot_rewards(all_rewards)
    
    final_checkpoint = {
        'episode': num_episodes - 1,
        'agents_state': [
            {
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                'obs_normalizer_state': {
                    'running_mean': agent.obs_normalizer.running_mean,
                    'running_var': agent.obs_normalizer.running_var,
                    'count': agent.obs_normalizer.count
                }
            }
            for agent in trainer.agents
        ],
        'final_avg_reward': avg_reward,
    }
    torch.save(final_checkpoint, 'ippo_norm_final_model.pt')
    
    pygame.quit()
    log_file.close()
    plot_rewards(all_rewards)

if __name__ == "__main__":
    main() 