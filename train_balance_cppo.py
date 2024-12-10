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

# 定义中央化Actor网络
class CentralizedActor(nn.Module):
    def __init__(self, obs_dim, action_dim, n_agents, hidden_dim=512):
        super(CentralizedActor, self).__init__()
        self.n_agents = n_agents
        total_obs_dim = obs_dim * n_agents
        total_action_dim = action_dim * n_agents
        
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(),
        )
        
        self.mean_layer = nn.Linear(hidden_dim // 2, total_action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, total_action_dim) - 0.5)
        
        # 使用正交初始化
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0)
    
    def forward(self, obs_list):
        # 将所有智能体的观察连接在一起
        joint_obs = torch.cat(obs_list, dim=1)
        features = self.net(joint_obs)
        mean = torch.tanh(self.mean_layer(features))
        std = torch.exp(self.log_std)
        return mean, std
    
    def get_actions(self, mean, std):
        # 为每个智能体生成动作
        dist = Normal(mean, std)
        actions = dist.sample()
        actions = torch.clamp(actions, -1.0, 1.0)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean().clamp(min=0)
        
        # 将动作分割给每个智能体
        action_dim = mean.shape[1] // self.n_agents
        actions_list = torch.split(actions, action_dim, dim=1)
        log_probs_list = torch.split(log_probs, action_dim, dim=1)
        
        return actions_list, log_probs_list, entropy

# 定义中央化Critic网络
class CentralizedCritic(nn.Module):
    def __init__(self, obs_dim, n_agents, hidden_dim=512):
        super(CentralizedCritic, self).__init__()
        total_obs_dim = obs_dim * n_agents
        
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim, hidden_dim),
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
    
    def forward(self, obs_list):
        # 将所有智能体的观察连接在一起
        joint_obs = torch.cat(obs_list, dim=1)
        return self.net(joint_obs)

# 定义CPPO训练器
class CPPOTrainer:
    def __init__(self, env, device="cpu"):
        self.env = env
        self.device = device
        self.n_agents = len(env.agents)
        
        # 获取观察空间和动作空间维度
        obs = env.reset()
        self.obs_dim = obs[0].shape[1]
        self.action_dim = env.agents[0].action_size
        
        print(f"观察空间维度: {self.obs_dim}")
        print(f"动作空间维度: {self.action_dim}")
        print(f"智能体数量: {self.n_agents}")
        
        # 创建中央化的Actor和Critic网络
        self.actor = CentralizedActor(self.obs_dim, self.action_dim, self.n_agents).to(device)
        self.critic = CentralizedCritic(self.obs_dim, self.n_agents).to(device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # PPO参数
        self.clip_param = 0.2
        self.ppo_epochs = 15
        self.num_mini_batches = 4
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.05
        self.max_grad_norm = 0.5
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
        # 经验缓冲区
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def select_actions(self, obs_list):
        with torch.no_grad():
            mean, std = self.actor(obs_list)
            actions_list, log_probs_list, entropy = self.actor.get_actions(mean, std)
            value = self.critic(obs_list)
        return actions_list, log_probs_list, value, entropy
    
    def store_transition(self, obs_list, actions_list, reward, value, log_probs_list, done):
        self.observations.append(obs_list)
        self.actions.append(actions_list)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_probs_list)
        self.dones.append(done)
    
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
        
        # 重塑张量以便于批处理
        advantages = advantages.reshape(-1, 1)
        returns = returns.reshape(-1, 1)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self):
        # 计算GAE
        advantages, returns = self.compute_gae()
        
        # 准备数据
        obs_batch = [torch.cat([obs[i] for obs in self.observations]) for i in range(self.n_agents)]
        actions_batch = [torch.cat([actions[i] for actions in self.actions]) for i in range(self.n_agents)]
        old_log_probs_batch = [torch.cat([log_probs[i] for log_probs in self.log_probs]) for i in range(self.n_agents)]
        
        # 多次更新
        total_actor_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        batch_size = len(self.observations) * self.rewards[0].shape[0]
        mini_batch_size = batch_size // self.num_mini_batches
        
        for _ in range(self.ppo_epochs):
            # 生成随机索引
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_indices = indices[start:end]
                
                # 获取mini-batch数据
                mb_obs = [obs[mb_indices] for obs in obs_batch]
                mb_actions = [actions[mb_indices] for actions in actions_batch]
                mb_old_log_probs = [old_log_probs[mb_indices] for old_log_probs in old_log_probs_batch]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # 计算新的动作分布
                mean, std = self.actor(mb_obs)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(torch.cat(mb_actions, dim=1))
                entropy = dist.entropy().mean().clamp(min=0)
                
                # 将log_probs分割回每个智能体
                new_log_probs_list = torch.split(new_log_probs, self.action_dim, dim=1)
                
                # 计算每个智能体的策略比率和损失
                ratios = [(torch.exp(new_log_probs - old_log_probs)) for new_log_probs, old_log_probs in zip(new_log_probs_list, mb_old_log_probs)]
                surr1 = [ratio * mb_advantages for ratio in ratios]
                surr2 = [torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages for ratio in ratios]
                actor_loss = -torch.mean(torch.stack([torch.min(s1, s2).mean() for s1, s2 in zip(surr1, surr2)]))
                
                # 计算值函数损失
                value_pred = self.critic(mb_obs)
                value_loss = 0.5 * ((mb_returns - value_pred) ** 2).mean()
                
                # 计算总损失
                loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # 更新Actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # 更新Critic
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        # 清空内存
        self.clear_memory()
        
        num_updates = self.ppo_epochs * (batch_size // mini_batch_size)
        return total_actor_loss / num_updates, total_value_loss / num_updates, total_entropy / num_updates
    
    def train_episode(self):
        obs = self.env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 收集动作
            actions_list, log_probs_list, value, _ = self.select_actions(obs)
            
            # 存储转换
            self.store_transition(
                obs,
                actions_list,
                torch.zeros(self.env.num_envs, 1, device=self.device),  # 初始奖励
                value,
                log_probs_list,
                torch.zeros(self.env.num_envs, 1, device=self.device)  # 初始done状态
            )
            
            # 执行动作
            next_obs, rewards, dones, _ = self.env.step(actions_list)
            done = any(d.any() for d in dones)
            
            # 更新奖励和done状态
            mean_reward = torch.stack([r.mean(dim=0) for r in rewards]).mean() * 100
            self.rewards[-1] = mean_reward.expand(self.env.num_envs, 1)
            self.dones[-1] = torch.full((self.env.num_envs, 1), float(done), device=self.device)
            episode_reward += mean_reward.item()
            
            obs = next_obs
        
        # 更新策略
        actor_loss, value_loss, entropy = self.update()
        
        return actor_loss, value_loss, entropy, episode_reward

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
    plt.savefig('cppo_training_rewards.png')
    plt.close()

def main():
    # 初始化环境
    env = vmas.make_env(
        scenario="balance",
        num_envs=16,
        device="cpu",
        continuous_actions=True,
        seed=0,
        n_agents=4
    )
    
    # 初始化CPPO训练器
    trainer = CPPOTrainer(env)
    
    # 训练参数
    num_episodes = 500  # 修改为500回合
    
    # 初始化Pygame
    pygame.init()
    frame = env.render(mode="rgb_array")
    height, width = frame.shape[:2]
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("VMAS Balance CPPO Training")
    clock = pygame.time.Clock()
    
    # 记录训练数据
    all_rewards = []
    episode_rewards = deque(maxlen=100)
    best_avg_reward = float('-inf')
    
    # 创建日志文件
    log_file = open('cppo_training_log.txt', 'w')
    
    # 渲染设置
    render_interval = 1  # 每隔多少episode渲染一次
    frame_rate = 60  # 目标帧率
    
    # 训练循环
    for episode in range(num_episodes):
        # 处理Pygame事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                log_file.close()
                return
        
        # 训练一个回合
        actor_loss, value_loss, entropy, episode_reward = trainer.train_episode()
        
        # 记录和显示训练进度
        all_rewards.append(episode_reward)
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards)
        
        # 更新最佳平均奖励
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            # 保存最佳模型
            checkpoint = {
                'episode': episode,
                'actor_state_dict': trainer.actor.state_dict(),
                'critic_state_dict': trainer.critic.state_dict(),
                'actor_optimizer_state_dict': trainer.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': trainer.critic_optimizer.state_dict(),
                'best_avg_reward': best_avg_reward,
            }
            torch.save(checkpoint, 'cppo_best_model.pt')
        
        # 记录训练信息
        log_info = f"Episode {episode + 1}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}\n"
        log_info += f"Actor Loss: {actor_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy: {entropy:.4f}\n"
        log_info += f"Best Avg Reward: {best_avg_reward:.2f}\n"
        
        # 打印到控制台和写入日志文件
        print(log_info)
        log_file.write(log_info + '\n')
        log_file.flush()  # 确保立即写入文件
        
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
                log_file.write(f"渲染错误: {e}\n")
        
        # 每100个episode绘制一次奖励曲线
        if (episode + 1) % 100 == 0:
            plot_rewards(all_rewards)
    
    # 保存最终模型
    final_checkpoint = {
        'episode': num_episodes - 1,
        'actor_state_dict': trainer.actor.state_dict(),
        'critic_state_dict': trainer.critic.state_dict(),
        'actor_optimizer_state_dict': trainer.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': trainer.critic_optimizer.state_dict(),
        'final_avg_reward': avg_reward,
    }
    torch.save(final_checkpoint, 'cppo_final_model.pt')
    
    pygame.quit()
    log_file.close()
    
    # 保存最终的奖励曲线
    plot_rewards(all_rewards)

if __name__ == "__main__":
    main() 