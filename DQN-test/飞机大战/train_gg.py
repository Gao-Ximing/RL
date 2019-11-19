
from pygame.color import THECOLORS
from env_gg import uav_env
import os
import pygame,sys
from math import *
import numpy as np
from scipy import io
# #solo RL
# from RL_brainbk import DuelingDQN
# mean field RL
from RL_brain import DuelingDQN

import tensorflow as tf
import matplotlib.pyplot as plt

game=True
if game:
    pygame.init()
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (50, 70)  # 设置窗口在屏幕的相对位置
    screen = pygame.display.set_mode((1800, 1000), 0, 32)

    redAgentImage0 = pygame.transform.scale(pygame.image.load('element/blue.png'), (16, 16))  # 缩放函数
    redAgentImage = pygame.transform.scale(pygame.image.load('element/red.png'), (16, 16)) # 缩放函数
    redAgentImage1= pygame.transform.scale(pygame.image.load('element/red_1.png'), (16, 16)) # 缩放函数
    blueAgentImage = pygame.transform.scale(pygame.image.load('element/aim.png'), (36, 36)) # 缩放函数

    clock = pygame.time.Clock()
    plot = True

def run_DQN(RL):

    episode = 0
    max_reward = 0
    #agentstate = np.zeros([env.agentN[2], 4])
    acc_r = None
    ave_rr = None
    max_step = 4001  # 2000   #最大步数
    reward_t = np.zeros([15, max_step])

    #大的训练轮数循环
    while episode < max_step:
        # initial observation
        env.reset()   #环境初始化
        agentstate=env.Agent   #（红+蓝）*6维矩阵，[x,y,theta,v,i, 0]，位置，角度，速度，编号
        train_step=0
        ep_reward=0    #回合奖励
        acc_reward = 0
        step=0
        rewardofi = np.zeros(env.redN)

        m_rand=np.random.randint(0,7)

        if episode%45==0:
            RL.epsilon=0.6

        #单回合训练循环
        while True:

            #环境显示等函数
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            if game:
                screen.fill((255, 255, 255))  #显示白色背景


            # 下面为整个更新计算的主要策略及步骤 显示相关程序的不介绍
            # 1 遍历整个红军个体 计算更新状态
            #   a 获取他们的当前状态
            #   b 根据状态选择动作
            #   c 当前状态和动作 计算下一状态
            # 2 根据更新的状态,更新计算矩阵,为后面计算做准备
            # 3 再次遍历整个红军 计算奖励
            #   a 获取更新后的状态
            #   b 根据状态计算奖励值
            #   c 保存所有的参数到经验池
            # 4 网络学习

            # 1 遍历红军个体
            for i in range(env.redN):
                #上一轮的Feature
                env.lastFeature[i] = env.feature[i]

                # a 获取他们的当前状态
                env.feature[i]=env.getfeature(i)  #获取个体i的状态
                #print(env.feature[i][25] , env.lastFeature[i][25])
                # RL choose action based on observation

                #   b 根据状态选择动作
                #动作选择策略 包括纯网络选择,加随机概率的网络选择,加固定步数纯随机选择的网络选择
                ##前13个按概率方法选择动作,后两个一直随机选择动作
                if i<2:
                    env.actions[i] = RL.choose_action(env.feature[i][0:env.n_features], 2)  # 纯网络模式选择动作
                elif i<11:
                    env.actions[i] = RL.choose_action(env.feature[i][0:env.n_features],-1)  #默认模式选择动作
                else:
                    if (step//50+i)%7==m_rand:
                        env.actions[i] = RL.choose_action(env.feature[i][0:env.n_features], i%2)  # 随机模式选择左右动作
                    else:
                        env.actions[i] = RL.choose_action(env.feature[i][0:env.n_features], 2)  # 纯网络模式选择动作


                # RL take action and get next observation and reward
                # 获取个体i新的状态，（红+蓝）*6维矩阵，[x,y,theta,v,i, 0]，位置，角度，速度，编号

                #   c 当前状态和动作 计算下一状态
                #agentstate[i][0:4], env.nextfeature[i], env.reward[i],env.sc[i] = env.step(env.actions[i], i)
                agentstate[i][0:4] = env.step(env.actions[i], i)

                # rewardofi[i] += env.reward[i]
                # #每个个体的奖赏统计
                # reward_t[i][episode] = 0.8*reward_t[i][episode-1] + 0.2*env.reward[i]

                x, y, t, v = agentstate[i][0:4] #读出状态值，用于界面显示
                if game:
                    #screen.blit(pygame.transform.rotate(redAgentImage, -t), (200+x / 5, y / 5))
                    if i<2:
                        screen.blit(pygame.transform.rotate(redAgentImage0, -t), (x - 8, y - 8))
                    elif i<11:
                        screen.blit(pygame.transform.rotate(redAgentImage, -t), (x-8, y-8))
                    else :
                        screen.blit(pygame.transform.rotate(redAgentImage1, -t), (x-8, y-8))


            if game: #显示蓝军
                #print(env.blueN)
                for i in range(env.blueN):
                    sss=env.Agent[i + env.redN]
                    screen.blit(pygame.transform.rotate(blueAgentImage, -sss[2]), (sss[0]-18, sss[1]-18))
                    pygame.draw.circle(screen, THECOLORS['blue'], (int(sss[0]), int(sss[1])), env.bluekillnum, 1)  # 目标被攻击距离
                    pygame.draw.circle(screen, THECOLORS['green'], (int(sss[0]), int(sss[1])), env.bluekillnum*10, 1)  # 目标消失时,飞机奖励惩罚范围
                    pygame.draw.circle(screen, THECOLORS['green'], (int(sss[0]), int(sss[1])), env.bluekillnum//2, 1)  # 单个飞机奖励距离
                    pygame.draw.circle(screen, THECOLORS['red'], (int(sss[0]), int(sss[1])), 10*i+10, 1)  # 标记目标的排名,没多大意义
                pygame.display.update()
            #ep_reward += np.mean(env.reward)

            # 2 根据更新的状态,更新计算矩阵,为后面计算做准备
            env.s2f()   #更新矩阵
            # 计算距离矩阵 R:agentNum*2 X agentNum*2
            # R：
            # 红红 | 红蓝
            # 蓝红 | 蓝蓝

            j=0
            redN=env.redN
            blueN=env.blueN
            step += 1
            # 3 再次遍历整个红军 计算奖励
            for i in range(redN):   #计算奖赏

                #   a 获取更新后的状态
                env.nextfeature[i]=env.getfeature(i)

                #   b 根据状态计算奖励值
                env.reward[i], env.sc[i] = env.reward_gg(i,0,step)

               # env.sc[i]-=step*0.02

                acc_reward += env.sc[i]
                ep_reward+=env.sc[i]/redN
                train_step+=1

                #   c 保存所有的参数到经验池
                if np.random.uniform() < 0.4 or env.sc[i]<0:    #通过条件调整经验池中各样本的比重
                    RL.store_transition(env.feature[i][0:env.n_features],\
                    env.actions[i],env.sc[i], env.nextfeature[i][0:env.n_features]) #保存经验池
                # 保存经验池，s, a, r, s_


            # 4 网络学习
            if (train_step > 1000 or episode > 0):   #每轮学习一次
                RL.learn()
            # for i in range(env.redN):
            #     print(i,'\n',env.surAgentIndex[i], '\n')
            #     print('before','\n', env.feature[i], '\n')
            #     if env.blueN == 1:
            #         print('after', '\n', env.feature[i], '\n')
            # if env.blueN == 1:
            #     print('NEXT')
            #reward/=10  # normalize to a range of (-1, 0)
            #print(env.Agent[24][5], env.Agent[25][5], env.Agent[26][5], env.Agent[27][5], env.Agent[28][5])
            for i in range(redN+blueN):
                if env.Agent[i][5]:     #Agent第6个状态如果为1，则减少个体数量
                    if i<redN:
                        env.redN -= 1
                    else:
                        env.blueN -= 1
                    #print('red remain', env.redN, 'blue remain', env.blueN)
                    continue
                else:
                    env.Agent[j]=env.Agent[i]
                    j+=1

            # 退出回合循环并打印参数
            if env.redN <5 or env.blueN<1 or train_step > 15000: #红方小于5个，蓝方小于1个，训练12000轮 退出循环
                #print(rewardofi)
                print('episode:[',episode,']reward:',ep_reward,'remaining:[',env.redN,',',env.blueN,']',
                      'UsedSteps', '[', step, ']')
                episode += 1
                #RL.epsilon=0.2+(0.9-0.2)*episode/max_step
                if acc_r==None:     #累计奖赏值
                    acc_r=[]
                    acc_r.append(ep_reward)
                else:
                    acc_r.append(ep_reward)#(ep_reward*0.1 + acc_r[-1] * 0.9)  # accumulated reward
                # 保存网络参数
                if (episode+1)%10 == 0:# or max_reward <= ep_reward: #每训练50的倍数或训练奖赏大于当前最大奖赏
                    max_reward = ep_reward
                    RL.stor_DQN(0)                       #保存网络
                    print('stor network episode:',episode)
                ave_r = 0
                #print(episode)
                if episode>20:
                    #print(episode)
                    #print(acc_r)
                    for ii in range(20):
                        ave_r += acc_r[episode-ii-1]
                    ave_r = ave_r/20
                    if ave_rr == None:  # 累计奖赏值
                        ave_rr = []
                        ave_rr.append(ave_r)
                    else:
                        ave_rr.append(ave_r)
                # break while loop when end of this episode
                break
            # 边界处理
            if env.redN >=5 and env.blueN >= 0:  #如果红方个体数大于5，蓝方个体数大于1
                env.boundworld()

    #print(ave_rr)
    RL.plot_cost()
    print('train over')
    #总奖赏
    plt.plot(np.array(acc_r), c='b', label='dueling')
    plt.savefig("rewards/reward.svg")
    plt.legend(loc='best')
    plt.ylabel('accumulated reward')
    plt.xlabel('training steps')
    plt.grid()
    plt.show()
    #个体奖赏
    #print(reward_t[0])
    plt.plot(np.array(reward_t[0]), color='#15b01a', label='dueling')
    plt.plot(np.array(reward_t[1]), color='r', label='dueling')
    plt.plot(np.array(reward_t[2]), color='b', label='dueling')
    plt.savefig("rewards/reward_t.svg")
    plt.legend(loc='best')
    plt.ylabel('reward for UAV')
    plt.xlabel('training steps')
    plt.grid()
    plt.show()
    #平均奖赏
    plt.plot(np.array(ave_rr), c='r', label='dueling')
    plt.legend(loc='best')
    plt.ylabel('average reward of 20 times')
    plt.xlabel('training steps')
    plt.grid()
    plt.savefig("rewards/ave_reward.svg")
    plt.show()
    #保存数据
    io.savemat('rewards/reward_t.mat', {'array': np.array(reward_t)})
    io.savemat('rewards/acc_r.mat', {'array': np.array(acc_r)})
    io.savemat('rewards/ave_rr.mat', {'array': np.array(ave_rr)})

if __name__ == "__main__":
    # uav battle game
    env = uav_env(15,5)        #15个红对5个目标的环境
    sess1 = tf.Session()        #定义会话对象
    with tf.variable_scope('dueling'):  #duelingDQN
        dueling_DQN = DuelingDQN(
            n_actions=env.n_actions, n_features=env.n_features, memory_size=50000,
            replace_target_iter=1000,e_greedy=0.7,
            e_greedy_increment=0.1, sess=sess1, dueling=True, output_graph=False)
        #动作空间维度，状态空间维度，记忆库，贪婪系数0.05，句柄，dueling 开，输出图 关

    sess1.run(tf.global_variables_initializer())  #网络初始化
    dueling_DQN.load_DQN(0)
    run_DQN(dueling_DQN)

    #RL.load_DQN()


