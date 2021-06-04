import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
#生成随机数据，2个高斯模型
def creat_data(N):
    global X                  
    X = np.zeros((N))       #初始化X为N列。1维数据，有N个样本，赋值 N = 100
    global E            
    E = np.zeros((N,2))     #期望第i个样本属于第j个模型的概率的期望
    global alpha            
    alpha = [0.4,0.6]       #初始化混合项系数，权重为0.4和0.6
    global u
    u = [0,20]              #初始化均值分别为 0和20
    global sigma
    sigma = [np.sqrt(10),np.sqrt(15)]          #标准差分别为根号10和根号15
    for i in range(N):
        if np.random.random(1) < 0.4:          #生成0-1之间随机数，权重为0.4和0.6的情况下分别取两个高斯分布的值
            X[i]  = np.random.normal(0, np.sqrt(10), None)      #用第一个高斯模型生成数据
        else :
            X[i] = np.random.normal(20, np.sqrt(15), None)      #用第二个高斯模型生成数据
    print("可观测数据：\n",X)       #输出可观测样本
    print("初始化的u1，u2:",u)      #输出初始化的mu
 
def E_Step(sigma,k,N):
    global X
    global E
    global alpha
    for i in range(N):
        fenmu=0
        for j in range(0,k):
            fenmu += alpha[j]*math.exp(-(X[i]-u[j])*sigma[j]*np.transpose(X[i]-u[j]))/np.sqrt(sigma[j])       #分母
        for j in range(0,k):
            fenzi = math.exp(-(X[i]-u[j])*sigma[j]*np.transpose(X[i]-u[j]))/np.sqrt(sigma[j])        #分子
            E[i,j]=alpha[j]*fenzi/fenmu      #求期望
    print("隐藏变量：\n",E)
 
def M_Step(k,N):
    global E
    global X
    global alpha
    for j in range(0,k):
        fenmu=0   #分母
        fenzi=0   #分子
        for i in range(N):
            fenzi += E[i,j]*X[i]
            fenmu += E[i,j]
        u[j] = fenzi/fenmu    #求均值
        alpha[j]=fenmu/N        #求混合项系数
 
if __name__ == '__main__':
    
    iter_num=5  #迭代次数
    N=100        #样本数目
    k=2           #高斯模型数
    probility = np.zeros(100)    #混合高斯分布
    # global sigma          #协方差矩阵
    creat_data(100)     #生成数据
    #迭代计算
    for i in range(iter_num):
        err=0     #均值误差
        err_alpha=0    #混合项系数误差
        old_u = copy.deepcopy(u)
        old_alpha = copy.deepcopy(alpha)
        E_Step(sigma,k,N)     # E步
        M_Step(k,N)           # M步
        print("迭代次数:",i+1)
        print("估计的均值:",u)
        print("估计的混合项系数:",alpha)
        for z in range(k):
            err += (abs(old_u[z]-u[z])+abs(old_u[z]-u[z]))      #计算误差
            err_alpha += abs(old_alpha[z]-alpha[z])
        if (err<=0.00001) and (err_alpha<0.00001):     #达到精度退出迭代
            print(err,err_alpha)
            break