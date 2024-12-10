import json
import numpy as np
import matplotlib.pyplot as plt

def plot_training_rewards(log_file='mappo_training_log.txt', window_size=100):
    # 读取日志文件
    episodes = []
    rewards = []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                episodes.append(data['episode'])
                rewards.append(data['reward'])
    except Exception as e:
        print(f"读取日志文件时出错: {e}")
        return
    
    # 转换为numpy数组
    rewards = np.array(rewards)
    
    # 计算移动平均
    def moving_average(data, window_size):
        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode='valid')
    
    avg_rewards = moving_average(rewards, window_size)
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 绘制原始奖励
    plt.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw Rewards')
    
    # 绘制移动平均
    # 由于移动平均会损失开始的window_size-1个点，需要调整x轴
    avg_episodes = episodes[window_size-1:]
    plt.plot(avg_episodes, avg_rewards, color='red', 
             label=f'Average over {window_size} episodes')
    
    # 设置图表属性
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 保存图表
    try:
        plt.savefig('mappo_training_rewards.png', dpi=300, bbox_inches='tight')
        print("图表已保存为 mappo_training_rewards.png")
    except Exception as e:
        print(f"保存图表时出错: {e}")
    
    plt.close()

if __name__ == "__main__":
    plot_training_rewards() 