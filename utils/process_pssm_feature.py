import numpy as np

def addToOneLineSum(pssm):
    return np.sum(sigmoid(pssm), 0)

def addToOneLineMean(pssm):
    return np.mean(sigmoid(pssm), 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 提取二维的PSSM特征
def getStandardPssm(pssm_dir):
    pssm = sigmoid(np.loadtxt(pssm_dir))
    # 设置两个数组保存L * 20 pssm矩阵的列均值和列方差
    mean = np.mean(pssm, axis=0) #每列的均值
    standard_deviation = np.sqrt(np.var(pssm, axis=0)) # 每列的方差
    content = np.zeros([20, 20])
    for s in range(20):
        for t in range(20):
            theta = 0
            for i in range(len(pssm) - 1):
                if standard_deviation[s] != 0 and standard_deviation[t] != 0:
                    M = (pssm[i, s] - mean[s]) * (pssm[i+1, t] - mean[t]) / (standard_deviation[s] * standard_deviation[t])
                elif standard_deviation[s] != 0:
                    M = (pssm[i, s] - mean[s]) * (pssm[i+1, t] - mean[t]) / standard_deviation[s]
                elif standard_deviation[t] != 0:
                    M = (pssm[i, s] - mean[s]) * (pssm[i+1, t] - mean[t]) / standard_deviation[t]
                else:
                    M = (pssm[i, s] - mean[s]) * (pssm[i+1, t] - mean[t])
                theta = theta + M
            content[s, t] = theta
    return content.reshape(400)