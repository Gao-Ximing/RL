"""
"""
# mean field RL
import numpy as np
import random
import heapq
from math import *
featindex = 0
class uav_env(object):
    def __init__(self,rednum=20 ,bluenum=3 ,x=1800,y=1000):
        super(uav_env, self).__init__()
        #空战环境范围 by defult 4000 x 2000
        self.wordsize=[x,y]
        #agent集群规模 红方+蓝方
        self.redN=rednum
        self.blueN=bluenum
        self.agentN = [rednum,bluenum,rednum+bluenum]
        self.bluekillnum=30
        #《动作空间》
        #self.action_space = ['l','r','t','attack']
        self.n_actions = 9#len(self.action_space)
        self.actions = np.zeros(self.agentN[2],dtype='int')

        self.sensorR = 80
        self.RedsensoredNum = np.zeros(self.redN)
        self.BluesensoredNum = np.zeros(self.blueN)
        self.RedsensoredNumindex = np.zeros([self.redN,self.blueN],dtype='float')#红色感知距离内距离蓝方的距离排名
        #《状态空间》-> 5个友方 x 5维相对状态 + 1个敌方 x 3维相对状态
        self.apperDim={'friendNum':2,'friendFeature':5,'enemyNum':2,'enemyFeature':3 }
        self.n_features = 2*5 + 3*2 + 5 #最后一个是可感知邻居个数 #7*(5)+2+1
        global featindex
        featindex=self.apperDim['friendFeature']* self.apperDim['friendNum']\
                  + self.apperDim['enemyNum']* self.apperDim['enemyFeature'] + 4
        #print(featindex)
            #在这里用"feature"作为DQN输入参数,state要转换为feature
        self.feature = np.zeros([self.agentN[2],self.n_features],dtype='float64')
        self.lastFeature = np.zeros([self.agentN[2],self.n_features],dtype='float64')
        self.lastDalpha = np.zeros(self.redN ,dtype='float64')
        self.nextfeature = np.zeros([self.agentN[2], self.n_features], dtype='float64')
            #state为agent邻域内5个agent的状态，敌方5个agent的状态，用距离R矩阵判断邻域
            #二维环境：<x,y,v_x,v_y>
            #三维环境：<x,y,z,v_x,v_y,v_z>
            #agent状态
        self.Agent=np.zeros([self.agentN[2],6])
        self.reward=np.zeros(self.agentN[2])
        self.sc = np.zeros(self.agentN[0])
        #self.blueAgent = np.zeros([self.redN, 5])
        #初始化距离矩阵
        self.rrD = [[0 for col in range(self.redN)]for row in range(self.redN)]
        self.rbD = [[0 for col in range(self.blueN)] for row in range(self.redN)]
        self.brD = [[0 for col in range(self.redN)] for row in range(self.blueN)]
        self.bbD = [[0 for col in range(self.blueN)] for row in range(self.blueN)]
        self.meanfield={'red':0,'blue':0}
        #初始化邻域感知范围
        self.surAgentIndex=np.zeros([self.agentN[2], self.apperDim['friendNum']+self.apperDim['enemyNum']],dtype=int)


    def reset(self):
        self.redN=self.agentN[0]
        self.blueN=self.agentN[1]
        self.x_base=random.randint(600,900)
        self.y_base=random.randint(600,800)
        #self.Agent=self.BlueAgent
        # x_base=(random.random()*400+200)
        # y_base=(random.random()*200+100)
        for i in range(self.agentN[0]):
            #红方agent索引范围:0~agentnum
            #红方agent状态初始化 限制范围在 x:0～200 y:0~2000
                #self.Agent[i*10+ii] = [random.randint(0,self.wordsize[0]/50)*50, random.randint(0,self.wordsize[1]/50)*50, \
                     #random.uniform(0, 1)*50,0,0,0]
            x = random.random() * 100+self.x_base #(random.random() * 100 + x_base )
            y = random.random() * 100+self.y_base #(random.random() * 100 + y_base )
            #y=i/self.agentN[0]*500
            theta = (random.random())*360
            v = (random.random()*20 + 10)
            self.Agent[i] = [x,y,theta,v,i, 0]
        theta = 0
        #左上角为原点,横x,纵y
        x = random.random() * 200 +100#300#((random.random()) * 300 ) * 5
        y = random.random() * 150+100#((random.random()) * 200 ) * 5
        self.Agent[0 + self.agentN[0]] = [x, y, theta, 30, 0 + self.agentN[0], 0]
        x = random.random() * 200 +100#300#((random.random()) * 300 ) * 5
        y = random.random() * 100+620#400#((random.random()) * 200 + 400 ) * 5
        self.Agent[1 + self.agentN[0]] = [x, y, theta, 30, 1 + self.agentN[0], 0]
        x = random.random() * 300+450#500#((random.random()) * 300 + 300) * 5
        y = random.random() * 150 +350#300#((random.random()) * 200 + 200) * 5
        self.Agent[2 + self.agentN[0]] = [x, y, theta, 30, 2 + self.agentN[0], 0]
        x = random.random() * 200+950#700#((random.random()) * 200 + 600) * 5
        y = random.random() * 100+100#200#((random.random()) * 200 + 300 ) * 5
        self.Agent[3 + self.agentN[0]] = [x, y, theta, 30, 3 + self.agentN[0], 0]
        x = random.random() * 300+1300#7700#((random.random()) * 200 + 600 ) * 5
        y = random.random() * 300+460#400#((random.random()) * 200 ) * 5
        self.Agent[4 + self.agentN[0]] = [x, y, theta, 30, 4 + self.agentN[0], 0]
        #
        # x = 1300#((random.random()) * 200 + 600 ) * 5
        # y = 600#((random.random()) * 200 ) * 5
        # self.Agent[5 + self.agentN[0]] = [x, y, theta, 30, 5 + self.agentN[0], 0]

        # for i in range(self.agentN[1]):
        #     x = ((random.random()) * 400 + i*100 ) * 5
        #     # y = (i-5)*100+random.randint(-50,50) + 1000
        #     y = ((random.random()) * 300 + i*50) * 5
        #     theta=0#(random.random())*360
        #     self.Agent[i+self.agentN[0]] = [x,y,theta,30,i+self.agentN[0],0]
        # # self.Agent[self.agentN[0]] = [3000, 1000, 0, 30, i + self.agentN[0], 0]
        self.s2f()

    def getfeature(self, whichagent): #与感知范围内其他个体的位置差，速度差，与各个目标的距离
        def calcufield(whichagent, targetagent, fore):  #输入：自身，所有邻居的索引向量，1表示友军
            deltax = self.Agent[targetagent][0] - self.Agent[whichagent][0]
            deltay = self.Agent[targetagent][1] - self.Agent[whichagent][1]
            dv = self.Agent[targetagent][3]-self.Agent[whichagent][3]
            # 计算距离 r
            r = sqrt(deltax ** 2 + deltay ** 2)
            # 角度转弧度
            alpha = radians(self.Agent[whichagent][2])
            belta = radians(self.Agent[targetagent][2])
            # v=self.Agent[targetagent][3]
            a = exp(-abs(r-50) / 1500)
            #b=v*0.025*a

            # 计算飞机与同伴的方位差
            # 计算目标方位
            targetAngDress = atan2(deltay, deltax)

            #targetAngDress = radians(targetAngDress)
            #targetAngDress = degrees(targetAngDress)
            # 计算方位差
            targetAngDress -= alpha
            while (targetAngDress > 2*pi):
                targetAngDress -= 2*pi
            while (targetAngDress < 0):
                targetAngDress += 2*pi

            if fore:
                #速度矢量夹角
                cossigma = (cos(alpha) * cos(belta) + sin(alpha) * sin(belta))
                field = cossigma*a*exp(-abs(dv))

                #计算速度角度差
                Angdv=alpha-belta
                while (Angdv > 2*pi):
                    Angdv -= 2*pi
                while (Angdv < 0):
                    Angdv += 2*pi

                #feature = [r / 2000, deltax / 2000, deltay / 2000, self.Agent[targetagent][2]/360, self.Agent[targetagent][3]/30, dv / 20, cossigma]
                feature = [r*10/sqrt(self.wordsize[0] ** 2 + self.wordsize[1] ** 2), targetAngDress, abs(dv)/10, Angdv,dv>0]        #距离 方位差 速度差 航向差
                return feature, field
            else:
                costheta = (cos(alpha) * deltax + sin(alpha) * deltay + 1e-10) / (r + 1e-10)
                cosgarma = (-cos(belta) * deltax - sin(belta) * deltay + 1e-10) / (r + 1e-10)
                field = (costheta-(cosgarma))/2*a+0.005*dv  #计算势场

                feature=[r*10/sqrt(self.wordsize[0] ** 2 + self.wordsize[1] ** 2), targetAngDress, 0]       #   距离  方位
                # feature = [r / 2000, deltax / 2000, deltay / 2000, costheta]
                return feature, field
        feature=np.zeros(self.n_features,dtype='float64')
        ii=0
        N=self.apperDim['friendNum']
        NN = self.apperDim['enemyNum']
        NNN = self.apperDim['friendFeature']
        NNNN = self.apperDim['enemyFeature']
        #print(NNNN)
        global featindex
        for index in self.surAgentIndex[whichagent][0:NN+N] : #邻域矩阵，总个体数x总邻居个数
            if ii<N: #对于友军
                o, a = calcufield(whichagent, index, 1) #输入：自身，所有邻居向量，表示友军
                feature[0 + ii * NNN:NNN + ii * NNN] = o #输出：友军的状态向量
                #feature[featindex]= self.RedsensoredNum[i] #最后一项是势场力，累加势场力的和
            else:   #对于敌军
                o, a = calcufield(whichagent, index, 0)
                #print(ii,0 + N * NNN + NNNN * (ii-N),NNNN +  N * NNN + NNNN * (ii-N),feature[0 + ii * NNN:NNNN + ii * NNN])
                feature[0 + N * NNN + NNNN * (ii-N) :NNNN +  N * NNN + NNNN * (ii-N)] = o
                #feature[featindex]-=a   #最后一项是势场力，累加势场力的和
            ii+=1
        # feature[(featindex-4):(featindex)] = np.array([self.Agent[whichagent][0]/1000,
        #                                                self.Agent[whichagent][1] / 600,
        #                                                self.Agent[whichagent][2] / 360,
        #                                                self.Agent[whichagent][4] / 30,])

        feature[(featindex - 4):(featindex)] = np.array([self.RedsensoredNumindex[whichagent][0]/10,
                                                         self.RedsensoredNumindex[whichagent][1]/10,
                                                         self.Agent[whichagent][3]/10,
                                                         self.blueN, ])# 0,])#

        feature[featindex] = self.RedsensoredNum[whichagent]
        #print(feature)
        return feature

    def s2f(self): # uav battle feature get from state
        # 计算距离矩阵 R:agentNum*2 X agentNum*2
        # R：
        # 红红 | 红蓝
        # 蓝红 | 蓝蓝
        # 红红
        #print(self.blueN)
        for i in range(self.redN):
            for j in range(i,self.redN):
                if i==j:
                    self.rrD[i][j]=float("inf")
                else:
                    self.rrD[i][j] = np.sqrt((self.Agent[i][0]-self.Agent[j][0])**2+(self.Agent[i][1]-self.Agent[j][1])**2)
                    self.rrD[j][i]=self.rrD[i][j]

        # print(self.Agent[24][5],self.Agent[25][5],self.Agent[26][5],self.Agent[27][5],self.Agent[28][5])
        #红蓝
        for i in range(self.redN):
            for j in range(self.blueN):
                if self.Agent[self.redN + j][5]:
                    self.rbD[i][j] = float("inf")
                else:
                    self.rbD[i][j] = np.sqrt((self.Agent[i][0] - self.Agent[j+self.redN][0]) ** 2 + (self.Agent[i][1] - self.Agent[j+self.redN][1]) ** 2)
                self.brD[j][i] = self.rbD[i][j]
        #计算红军可感知到的邻居个数
        self.RedsensoredNum = np.zeros(self.redN)
        for i in range(self.redN):
            for j in range(i,self.redN):
                if self.rrD[i][j] < self.sensorR :
                    self.RedsensoredNum[i] += 1

        #计算蓝军可感知到的红军个数
        self.BluesensoredNum = np.zeros(self.blueN)
        for j in range(self.blueN):
            for i in range(self.redN):
                #print(self.rbD[i][j])
                if self.rbD[i][j] < self.bluekillnum :
                    self.BluesensoredNum[j] += 1
        #print(self.BluesensoredNum)
        N = self.apperDim['friendNum']
        NN = self.apperDim['enemyNum']
        for i in range(self.redN):
            #红方agent邻域内的agent索引
            #红红
            #在集合中找到最小的N个元素，然后找到与最小的几个元素对应的索引
            index=map(self.rrD[i][0:self.redN].index,heapq.nsmallest(N,self.rrD[i][0:self.redN]))
            self.surAgentIndex[i][0:N]=list(index) #计算得到距离最近的N个邻居的索引
            #红蓝
            index = map(self.rbD[i][0:self.blueN].index, heapq.nsmallest(NN, self.rbD[i][0:self.blueN]))
            self.surAgentIndex[i][N:N+NN] = [tmp+self.redN for tmp in list(index)]
        for i in range(self.blueN):
            # #蓝方agent邻域内的agent索引
            # #蓝蓝
            # index=map(self.bbD[i][0:self.blueN].index,heapq.nsmallest(N,self.bbD[i][0:self.blueN]))
            # self.surAgentIndex[i+self.redN][0:N]=[tmp+self.redN for tmp in list(index)]
            # #蓝红
            index = map(self.brD[i][0:self.redN].index, heapq.nsmallest(NN, self.brD[i][0:self.redN]))
            self.surAgentIndex[i+self.redN][0:NN]=list(index)
        # for i in range(self.redN):
        #     self.feature[i]=self.getfeature(i)

        # 计算红军可感知到的邻居个数中离蓝军距离的排名
        self.RedsensoredNumindex = np.zeros([self.redN,NN],dtype='float')
        for i in range(self.redN):
            for j in range(i, self.redN):
                if 1:#self.rrD[i][j] < self.sensorR:
                    for k in range(NN):
                        if self.rbD[j][self.surAgentIndex[i][N+k]-self.redN]<self.rbD[i][self.surAgentIndex[i][N+k]-self.redN]:
                            self.RedsensoredNumindex[i][k]+=1


    def step(self, action, whichagent):
        def judjekill(whichagent,n):
            enemy = self.surAgentIndex[whichagent][self.apperDim['friendNum']+n]
            # friend = self.surAgentIndex[whichagent][0]
            if whichagent >= self.redN:
                r = self.brD[whichagent - self.redN][enemy]
                # rr=self.bbD[whichagent-self.redN][friend-self.redN]
            else:
                r = self.rbD[whichagent][enemy - self.redN]
                # rr = self.bbD[whichagent][friend]
            global featindex
            if (50/5 <r< 100/5):
                alpha = radians(self.Agent[enemy][2]) #角度转换为弧度
                #sigma = radians(self.Agent[whichagent][2])
                deltax = self.Agent[enemy][0] - self.Agent[whichagent][0]
                deltay = self.Agent[enemy][1] - self.Agent[whichagent][1]
                costheta = (cos(alpha) * deltax + sin(alpha) * deltay) / (r + 1e-7)
                #cosgarma=(cos(alpha) * cos(sigma) + sin(alpha) * sin(sigma))
                if self.feature[whichagent][featindex]>0:
                    if costheta>0.8 and self.feature[whichagent][featindex]>random.random():
                    #if random.random() > 0.01:
                        self.Agent[enemy][5] = 1
                        #print('agent', whichagent, 'kill enemy', enemy)
                        return True
                    else:
                        return False
                else:
                    self.Agent[whichagent][5] = 1
                    return False
            else:
                return False
        x=self.Agent[whichagent][0]
        y=self.Agent[whichagent][1]
        alpha=self.Agent[whichagent][2]
        v = self.Agent[whichagent][3]
        reward = 0
        # 左加速
        if action == 0:
            alpha -= 5
            v += 5
            reward -= 0.1
        #左保持
        elif action == 1:
            alpha -= 5
            v -= 0
            reward -= 0.1
        # 左减速
        elif action == 2:
            alpha -= 6
            v -= 1
            reward -= 0.1
        #直减
        elif action == 3:
            v -= 1
            reward -= 0.1
        #保持
        elif action == 4:
            v -= 0
            reward -= 0.1
        #加速
        elif action == 5:
            v += 5
            reward -= 0.1
        #右减速
        elif action == 6:
            alpha += 6
            v -= 1
            reward -= 0.1
        # 右保持
        elif action == 7:
            alpha += 5
            v += 0
            reward -= 0.1
        # 右加速
        else:
            alpha += 5
            v += 5
            reward -= 0.1

        if alpha>360:
            alpha = alpha-360
        if alpha<0:
            alpha=alpha+360
        if v>60:
            v=60
        elif v<10:
            v=10
        x+= ((v*(cos(radians(alpha))))/10)
        y+= ((v*(sin(radians(alpha))))/10)
        #print(v)
        #计算奖赏
        #更新状态，计算feature
        self.Agent[whichagent][0:4]=[x,y,alpha,v]
        #f=self.getfeature(whichagent)


        return [x,y,alpha,v]  #,f,reward       #返回自身信息,特征,临时回报值

    def reward_gg(self, whichagent,reward,step):

        sc = reward

        # ave_alpha = sum(self.Agent[self.surAgentIndex[whichagent][0:5]][2])/5
        # Dalpha = alpha - ave_alpha
        # if Dalpha - self.lastDalpha[whichagent] < 0:
        #     reward += 0.0
        # else:
        #     reward -= 0.0
        # self.lastDalpha[whichagent] = Dalpha
        # #计算接近目标的reward
        # dr1 = self.feature[whichagent][10] - self.lastFeature[whichagent][10]
        # dr2 = self.feature[whichagent][13] - self.lastFeature[whichagent][13]
        # #print(100*dr1, 100*dr2)
        # #威力场势函数奖赏
        # #print(dr1,dr2)

        attack = 0
        for i in range(self.blueN):
            if self.rbD[whichagent][i] <= 50:
                attack += 1
        sec = 0

        # 目标方向奖励,可以引导飞机飞向目标
        if abs(self.nextfeature[whichagent][11]-pi)> pi*6/7:        #与第一个目标角度小于一个度数奖励
            sec+=0
        elif abs(self.nextfeature[whichagent][14] - pi)>pi*6/7:     #与第二个目标角度小于一个度数奖励
            sec +=0
        else:               #否则给惩罚或着不惩罚
            sec +=-0

        # 探测距离有队友 奖励
        if self.nextfeature[whichagent][20]>0:
            sec+=0

        # 着三个动作都是加速的动作 奖励可提高飞行速度
        if self.actions[whichagent]==5 or self.actions[whichagent]==0 or self.actions[whichagent]==8:
            sec+=0

        # #停飞机
        # if self.blueN == 0:
        #     if self.Agent[whichagent][0]>1200 and self.Agent[whichagent][0]<1400
        #         and self.Agent[whichagent][1]>700 and  self.Agent[whichagent][1]<800 :
        #         sec+=10

        for i in range(self.blueN):
            # #远离靠近奖惩
            # if self.rbD[whichagent][i] >= 50 and attack == 0:
            #     if dr1 < -0.0001 :#or dr2 < -0.0001:
            #         reward += 4
            #     if dr1 >= -0.0001 and dr2 >= -0.0001 :
            #         reward -= 5
            # if self.rbD[whichagent][i] < 50 :
            #     if dr1 < -0.0001 :#or dr2 < -0.0001:
            #         reward += 1
            # #距离太远惩罚
            # if self.rbD[whichagent][i] >= 150 and attack == 0:
            #     reward -= 3
            # x, y = self.Agent[whichagent][0:2]
            # if x < 10 or x > self.wordsize[0] - 10 or y < 10 or y > self.wordsize[1] - 10:
            #     reward -= 3
            #
            # if self.rbD[whichagent][i] < 50:
            #     reward += 5
            # if self.BluesensoredNum[i] > 1 and self.rbD[whichagent][i] < 50:
            #     self.Agent[self.redN + i][5] = 1
            #     if self.BluesensoredNum[i] > 3:
            #         reward += 15
            #         sc += 15
            #     else:
            #         reward += ((5*self.BluesensoredNum[i]))
            #         sc += ((5*self.BluesensoredNum[i]))

            #计算奖罚--gg

            #当目标的被攻击范围内只有一个飞机的时候,这个飞机奖励,这样右单独的飞机离队,找到目标也会有奖励
            #未来可用来添加侦查功能
            if self.BluesensoredNum[i] == 1 and self.rbD[whichagent][i] < self.bluekillnum/2:
                sec += 10
            # 当目标范围内有多于n个飞机时候,目标消失,此时一定范围内的飞机按距离的排序安排奖励和惩罚
            if self.BluesensoredNum[i] > 1and self.rbD[whichagent][i] < self.bluekillnum*10:
                self.Agent[self.redN + i][5] = 1

                # 计算排序
                index=0
                for k in range(0,self.redN):
                    if self.rbD[whichagent][i]>self.rbD[k][i]:
                        index+=1
                #按照运行步数给奖励,步数越少奖励越大 并限幅
                sec+=30-step/20
                if sec<5:
                    sec=5

                # 按照排序安排奖励和惩罚,排名靠前奖励大,排名大于一定数给惩罚,为了让飞机分散
                if index==0:
                    sec+=32
                elif index==1:
                    sec+=30
                elif index==2:
                    sec+=10
                elif index==3:
                    sec+=5
                elif index==4:
                    sec +=0
                else:
                    alphy=atan2(self.Agent[i+self.redN][1] - self.Agent[whichagent][1],self.Agent[i+self.redN][0] - self.Agent[whichagent][0])
                    alphy-= radians(self.Agent[whichagent][2])

                    while alphy>pi:
                        alphy-=2*pi

                    while alphy <-pi:
                        alphy += 2 * pi

                    if -pi/4 < alphy < pi/4:
                        sec=-300
                    else:
                        sec=0




        return reward,sec #状态，特征（observation），动作奖赏，状态奖赏（存活奖赏）

    def ifattack(self,whichagent,action):
        def judjekill(whichagent, n):
            enemy = self.surAgentIndex[whichagent][self.apperDim['friendNum'] + n]
            # friend = self.surAgentIndex[whichagent][0]
            if whichagent >= self.redN:
                r = self.brD[whichagent - self.redN][enemy]
                # rr=self.bbD[whichagent-self.redN][friend-self.redN]
            else:
                r = self.rbD[whichagent][enemy - self.redN]
                # rr = self.bbD[whichagent][friend]
            if (100/5 < r < 150/5):
                self.Agent[enemy][5] = 1
        if action >= 7:
            judjekill(whichagent, action - 7)

    def steps(self, action, whichagent):

        x=self.Agent[whichagent][0]
        y=self.Agent[whichagent][1]
        alpha=self.Agent[whichagent][2]
        v = self.Agent[whichagent][3]
        reward = 0
        #直飞
        if action == 0:
            v+=0.01
            # reward -= 0.01
        #左飞
        elif action == 1:
            v-=0.01
            # reward -= 0.005
        # 右飞
        elif action == 2:
            alpha-=0.05
            reward -= 0.005
        elif action == 3:
            alpha+=0.05
            reward -= 0.005
        #左飞
        elif action == 4:
            alpha-=0.1
            reward -= 0.01
        # 右飞
        elif action == 5:
            alpha+=0.1
            reward -= 0.01
        elif action == 6:
            alpha+=0
            v+=0
        else:
            v+=0

        if alpha>360:
            alpha = alpha-360
        if alpha<0:
            alpha=alpha+360

        if v>30:
            v=30
        elif v<10:
            v=10
        x+= (v*(cos(radians(alpha)))/200)
        y+= (v*(sin(radians(alpha)))/200)
        #print(v,'\n')

        # ii=self.surAgentIndex[whichagent][4]
        # r=(self.rbD[whichagent][ii-self.redN])
        # delta_r=r-np.sqrt((self.Agent[ii][0]-x)**2+(self.Agent[ii][1]-y)**2)
        # reward+=delta_r*0.001

        #更新状态，计算feature
        self.Agent[whichagent][0:4]=[x,y,alpha,v]
        # f=self.getfeature(whichagent)

        return [x,y,alpha,v]

    def boundworld(self):
        for i in range(self.redN+self.blueN):
            x,y=self.Agent[i][0:2]
            if x < 5:
                x = 5
            elif x > self.wordsize[0] -5:
                x = self.wordsize[0] -5
            if y < 5:
                y = 5#(random.random()) * 600
            elif y > self.wordsize[1] -5:
                y = self.wordsize[1] -5
            self.Agent[i][0:2]=x,y

    def killred(self):
        for i in range(self.blueN):
            min=self.surAgentIndex[i+self.redN][0]
            if self.brD[i][min]<200:
                self.Agent[min][5]=1
