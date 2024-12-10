import vmas
import torch
import pygame
import numpy as np
import time

def main():
    # 初始化Pygame
    pygame.init()
    
    # 创建环境
    env = vmas.make_env(
        scenario="balance",    # 使用balance场景
        num_envs=32,          # 32个并行环境
        device="cpu",         # 使用CPU
        continuous_actions=True,  # 使用连续动作空间
        seed=0,              # 设置随机种子
        n_agents=4           # 4个智能体
    )
    
    print("环境创建成功!")
    print(f"智能体数量: {len(env.agents)}")
    
    # 重置环境
    obs = env.reset()
    print("环境重置成功!")
    print(f"观察空间大小: {len(obs)}")
    
    # 获取第一帧来设置窗口大小
    frame = env.render(mode="rgb_array")
    height, width = frame.shape[:2]
    
    # 创建Pygame窗口
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("VMAS Balance 场景")
    clock = pygame.time.Clock()
    
    running = True
    step = 0
    
    while running and step < 1000:  # 设置更长的运行时间
        step += 1
        print(f"\n步骤 {step}")
        
        # 处理Pygame事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # 获取随机动作
        actions = []
        for agent in env.agents:
            action = env.get_random_action(agent)
            actions.append(action)
        
        # 执行动作
        obs, rewards, dones, info = env.step(actions)
        
        print(f"奖励: {[reward.mean().item() for reward in rewards]}")
        
        # 渲染环境并显示
        try:
            frame = env.render(mode="rgb_array")
            # 将numpy数组转换为Pygame surface
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            
            # 控制帧率
            clock.tick(30)  # 30 FPS
            
        except Exception as e:
            print(f"渲染失败: {e}")
            break
        
        if any(done.any() for done in dones):
            print("环境结束!")
            # 等待几秒后关闭
            time.sleep(2)
            break
    
    print("\n测试完成!")
    pygame.quit()

if __name__ == "__main__":
    main() 