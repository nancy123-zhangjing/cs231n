from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax 损失函数，朴素实现（使用循环）

    输入维度为 D，共有 C 个类别，我们对大小为 N 的小批量数据进行操作。

    输入：
    - W: 一个形状为 (D, C) 的 NumPy 数组，包含权重。
    - X: 一个形状为 (N, D) 的 NumPy 数组，包含一个小批量数据。
    - y: 一个形状为 (N,) 的 NumPy 数组，包含训练标签；y[i] = c 表示 X[i] 的标签为 c，
        其中 0 <= c < C。
        也就是说Y记录的是n个图片每张图片属于哪一类

    - reg: (float) 正则化强度

    返回一个元组：
    - 损失值，一个浮点数
    - 相对于权重 W 的梯度；与 W 形状相同的数组
    """
    # 将损失和梯度初始化为零
    loss = 0.0
    dW = np.zeros_like(W)

    # 计算损失和梯度
    num_classes = W.shape[1]#类别数
    num_train = X.shape[0]
    for i in range(num_train): #图片数
        scores = X[i].dot(W)  # 计算得分

        # 以数值稳定的方式计算概率
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # 归一化
        logp = np.log(p)#这是一个(c,)的向量
        loss -= logp[y[i]]  # 负对数概率即为损失
        #现在已经得到了一张图片的预测向量，也就是每个种类的得分
        #我们现在还需要比对y[i]来决定梯度的计算
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += (p[j] - 1) * X[i]
            else:
                dW[:, j] += (p[j]) * X[i]
    dW /= num_train
    dW += 2*reg*W
    # 归一化后的合页损失加上正则化项
    loss = loss / num_train + reg * np.sum(W * W)
    
    #############################################################################
    # 代办:                                                                     #
    # 计算损失函数的梯度，并将其存储在 dW 中。                                    #
    # 而不是先计算损失，然后计算导数，可能更简单的方法是同时计算导数。              #
    # 因此，你可能需要修改上面的一些代码来计算梯度。                               #
    #############################################################################
    
    #############################################################################
    #                             你的代码结束                                   #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax 损失函数，向量化版本。

    输入和输出与 softmax_loss_naive 相同。
    """
    # 将损失和梯度初始化为零
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # 代办:                                                                     #
    # 实现 Softmax 损失的向量化版本，将结果存储在 loss 中。                        #
    #############################################################################
    xw = X @ W
    xw_max = np.max(xw,axis=1,keepdims=True)
    xw_exp = np.exp(xw-xw_max)
    xw_exp_sum = np.sum(xw_exp,axis=1,keepdims=True)
    xw_prop = xw_exp / xw_exp_sum #得到了所有图片的属于哪个类别的可能性矩阵，(N,C)
    xw_log = -np.log(xw_prop)
    n = xw.shape[0] #N
    num_classes = W.shape[1] 
    one_hot = np.zeros((n,num_classes))
    row_index = np.arange(n)
    col_index = y
    #用到高级索引的方式编码独热矩阵
    one_hot[row_index,col_index] = 1
    #使用高级索引取出loss
    loss = xw_log[row_index,col_index]
    dW = X.T@(xw_prop - one_hot)/n + 2*reg*W

    loss_final = sum(loss) / n + reg * np.sum(W * W)


    #############################################################################
    #                             你的代码结束                                   #
    #############################################################################

    #############################################################################
    # 代办:                                                                     #
    # 实现 Softmax 损失的梯度的向量化版本，将结果存储在 dW 中。                    #
    # 提示：与其从头开始计算梯度，不如重用一些计算损失时的中间值。                  #
    #############################################################################
    
    #############################################################################
    #                             你的代码结束                                  #
    #############################################################################

    return loss_final, dW
