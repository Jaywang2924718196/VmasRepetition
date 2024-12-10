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
import os
import json
from datetime import datetime

# 定义Actor网络
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
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim) - 1.0)
        
        # 使用正交初始化
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0)
        
    def forward(self, obs):
        features = self.net(obs)
        action_mean = torch.tanh(self.mean_layer(features))
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return action_mean, action_std

# 定义Critic网络
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
        
        # 使用正交初始化
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, obs):
        return self.net(obs)

# 定义MAPPO智能体
class MAPPOAgent:
    def __init__(self, obs_dim, action_dim, device="cpu"):
        self.device = device
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(obs_dim).to(device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3, eps=1e-5)
        
        # PPO参数
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        self.ppo_epoch = 4
        self.batch_size = 64
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        
        # 奖励归一化
        self.reward_normalizer = deque(maxlen=100)
        
    def get_action(self, obs):
        with torch.no_grad():
            mean, std = self.actor(obs)
            dist = Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob
    
    def evaluate_actions(self, obs, actions):
        with torch.set_grad_enabled(True):
            mean, std = self.actor(obs)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1).mean()
            return log_probs, entropy

    def update(self, observations, actions, old_log_probs, returns, advantages):
        total_actor_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.ppo_epoch):
            # Actor更新
            log_probs, entropy = self.evaluate_actions(observations, actions)
            ratio = torch.exp(log_probs - old_log_probs.detach())
            
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages.detach()
            
            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -self.entropy_coef * entropy
            
            # 优化Actor
            self.actor_optimizer.zero_grad()
            actor_total_loss = actor_loss + entropy_loss
            actor_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            # Critic更新
            value_pred = self.critic(observations.detach())
            value_loss = 0.5 * ((returns.detach() - value_pred) ** 2).mean()
            
            # 优化Critic
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
        
        # 返回平均损失
        n_updates = self.ppo_epoch
        return total_actor_loss / n_updates, total_value_loss / n_updates, total_entropy / n_updates

# 定义MAPPO训练器
class MAPPOTrainer:
    def __init__(self, env, device="cpu"):
        self.env = env
        self.device = device
        self.n_agents = len(env.agents)
        
        # 获取观察空间和动作空间维度
        obs = env.reset()  # 获取初始观察
        self.obs_dim = obs[0].shape[1]  # 每个智能体的观察维度
        self.action_dim = env.agents[0].action_size  # 每个智能体的动作维度
        
        print(f"观察空间维度: {self.obs_dim}")
        print(f"动作空间维度: {self.action_dim}")
        print(f"智能体数量: {self.n_agents}")
        
        # 创建智能体
        self.agents = [MAPPOAgent(self.obs_dim, self.action_dim, device) for _ in range(self.n_agents)]
        
        # 训练参数
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
    def collect_trajectories(self, max_steps):
        observations = [[] for _ in range(self.n_agents)]
        actions = [[] for _ in range(self.n_agents)]
        rewards = [[] for _ in range(self.n_agents)]
        values = [[] for _ in range(self.n_agents)]
        log_probs = [[] for _ in range(self.n_agents)]
        
        obs = self.env.reset()
        episode_reward = 0
        
        # 渲染设置
        frame_rate = 60
        last_render_time = time.time()
        render_interval = 1.0 / frame_rate
        
        for step in range(max_steps):
            current_time = time.time()
            should_render = current_time - last_render_time >= render_interval
            
            # 收集每个智能体的动作
            actions_list = []
            for i, agent in enumerate(self.agents):
                obs_tensor = obs[i]  # [N, obs_dim]
                action, log_prob = agent.get_action(obs_tensor)  # [N, action_dim], [N, 1]
                value = agent.critic(obs_tensor)  # [N, 1]
                
                observations[i].append(obs_tensor)
                actions[i].append(action)
                log_probs[i].append(log_prob)
                values[i].append(value)
                actions_list.append(action)
            
            # 执行动作
            next_obs, rewards_list, dones, _ = self.env.step(actions_list)
            
            # 存储奖励
            for i in range(self.n_agents):
                rewards[i].append(rewards_list[i])  # [N, 1]
                episode_reward += rewards_list[i].mean().item()
            
            # 更新观察
            obs = next_obs
            
            # 控制渲染频率
            if should_render:
                last_render_time = current_time
            
            if any(d.any() for d in dones):
                break
        
        # 转换为tensor并确保维度正确
        processed_data = []
        for i in range(self.n_agents):
            agent_data = {
                'observations': torch.stack(observations[i]),  # [T, N, obs_dim]
                'actions': torch.stack(actions[i]),  # [T, N, action_dim]
                'rewards': torch.stack(rewards[i]),  # [T, N, 1]
                'values': torch.stack(values[i]),  # [T, N, 1]
                'log_probs': torch.stack(log_probs[i])  # [T, N, 1]
            }
            processed_data.append(agent_data)
        
        return processed_data, episode_reward
    
    def compute_advantages(self, rewards, values, agent_idx):
        # 确保所有输入都是正确的维度
        if len(rewards.shape) == 3:  # [T, N, 1]
            rewards = rewards.squeeze(-1)  # [T, N]
        if len(values.shape) == 3:  # [T, N, 1]
            values = values.squeeze(-1)  # [T, N]
            
        T, N = rewards.shape  # T: 时间步长, N: 并行环境数
        advantages = torch.zeros_like(rewards)  # [T, N]
        returns = torch.zeros_like(rewards)  # [T, N]
        
        last_gae_lam = torch.zeros(N, device=rewards.device)  # [N]
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = torch.zeros(N, device=rewards.device)  # [N]
            else:
                next_value = values[t + 1]  # [N]
            
            delta = rewards[t] + self.gamma * next_value - values[t]  # [N]
            last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam  # [N]
            advantages[t] = last_gae_lam
            returns[t] = advantages[t] + values[t]
        
        # 标准化优势
        advantages = (advantages - advantages.mean(dim=0, keepdim=True)) / (advantages.std(dim=0, keepdim=True) + 1e-8)
        
        # 添加最后一个维度以匹配网络输出
        advantages = advantages.unsqueeze(-1)  # [T, N, 1]
        returns = returns.unsqueeze(-1)  # [T, N, 1]
        
        return advantages, returns
    
    def train_step(self):
        max_steps = 200
        trajectories, episode_reward = self.collect_trajectories(max_steps)
        
        # 更新每个智能体
        actor_losses = []
        value_losses = []
        entropies = []
        
        for i in range(self.n_agents):
            traj = trajectories[i]
            advantages, returns = self.compute_advantages(
                traj['rewards'],
                traj['values'],
                i
            )
            actor_loss, value_loss, entropy = self.agents[i].update(
                traj['observations'],
                traj['actions'],
                traj['log_probs'],
                returns,
                advantages
            )
            actor_losses.append(actor_loss)
            value_losses.append(value_loss)
            entropies.append(entropy)
        
        return np.mean(actor_losses), np.mean(value_losses), np.mean(entropies), episode_reward

def plot_rewards(rewards, window_size=100):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.4, label='Raw Rewards')
    plt.plot(np.convolve(rewards, np.ones(window_size)/window_size, mode='valid'), 
             label=f'Average over {window_size} episodes')
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('mappo_training_rewards.png')
    plt.close()

def main():
    # 创建保存目录
    try:
        # 确保当前目录存在并可写
        current_dir = os.getcwd()
        log_file = os.path.join(current_dir, "mappo_training_log.txt")
        
        # 清空已存在的日志文件
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("")  # 创建空文件
    except Exception as e:
        print(f"创建日志文件时出错: {e}")
        return
    
    # 初始化环境
    env = vmas.make_env(
        scenario="balance",
        num_envs=16,
        device="cpu",
        continuous_actions=True,
        seed=0,
        n_agents=4
    )
    
    # 初始化MAPPO
    trainer = MAPPOTrainer(env)
    
    # 训练参数
    num_episodes = 500
    
    # 初始化Pygame
    pygame.init()
    frame = env.render(mode="rgb_array")
    height, width = frame.shape[:2]
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("VMAS Balance MAPPO Training")
    clock = pygame.time.Clock()
    
    # 记录训练数据
    all_rewards = []
    episode_rewards = deque(maxlen=100)
    best_avg_reward = float('-inf')
    
    # 渲染设置
    render_interval = 1
    frame_rate = 60
    
    def save_checkpoint(episode, avg_reward, is_best=False, is_final=False):
        try:
            checkpoint = {
                'episode': episode,
                'agents_state': [agent.actor.state_dict() for agent in trainer.agents],
                'critics_state': [agent.critic.state_dict() for agent in trainer.agents],
                'avg_reward': avg_reward
            }
            
            if is_best:
                # 保存最佳模型
                model_path = os.path.join(current_dir, 'mappo_best_model.pth')
                torch.save(checkpoint, model_path)
                print(f"保存最佳模型到: {model_path}")
            
            if is_final:
                # 保存最终模型
                model_path = os.path.join(current_dir, 'mappo_final_model.pth')
                torch.save(checkpoint, model_path)
                print(f"保存最终模型到: {model_path}")
        except Exception as e:
            print(f"保存模型时出错: {e}")
    
    # 训练循环
    for episode in range(num_episodes):
        # 处理Pygame事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # 训练一个回合
        actor_loss, value_loss, entropy, episode_reward = trainer.train_step()
        
        # 记录和显示训练进度
        all_rewards.append(episode_reward)
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards)
        
        # 更新最佳平均奖励并保存最佳模型
        is_best = False
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            is_best = True
            save_checkpoint(episode + 1, avg_reward, is_best=True)
        
        # 记录训练日志
        log_info = {
            'episode': episode + 1,
            'reward': float(episode_reward),  # 确保数值可以被JSON序列化
            'avg_reward': float(avg_reward),
            'actor_loss': float(actor_loss),
            'value_loss': float(value_loss),
            'entropy': float(entropy),
            'best_avg_reward': float(best_avg_reward)
        }
        
        # 打印到控制台
        print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")
        print(f"Actor Loss: {actor_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy: {entropy:.4f}")
        print(f"Best Avg Reward: {best_avg_reward:.2f}")
        
        # 保存到日志文件
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_info) + '\n')
        except Exception as e:
            print(f"写入日志时出错: {e}")
        
        # 渲染
        if episode % render_interval == 0:
            try:
                frame = env.render(mode="rgb_array")
                surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                screen.blit(surface, (0, 0))
                pygame.display.flip()
                clock.tick(frame_rate)
            except Exception as e:
                print(f"渲染错误: {e}")
        
        # 每100个episode绘制一次奖励曲线
        if (episode + 1) % 100 == 0:
            plot_rewards(all_rewards)
            plt.savefig('mappo_training_rewards.png')
    
    pygame.quit()
    
    # 保存最终的奖励曲线
    plot_rewards(all_rewards)
    plt.savefig('mappo_training_rewards.png')
    
    # 保存最终模型
    save_checkpoint(num_episodes, avg_reward, is_final=True)

if __name__ == "__main__":
    main() 