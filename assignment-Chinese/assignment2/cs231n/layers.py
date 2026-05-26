from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    计算仿射（全连接）层的前向传播。

    输入x的形状为(N, d_1, ..., d_k)，包含N个样本的小批量数据，
    其中每个样本x[i]的形状为(d_1, ..., d_k)。我们会将每个输入重塑为
    维度为D = d_1 * ... * d_k的向量，然后将其转换为维度为M的输出向量。

    输入：
    - x: 包含输入数据的numpy数组，形状为(N, d_1, ..., d_k)
    - w: 权重的numpy数组，形状为(D, M)
    - b: 偏置的numpy数组，形状为(M,)

    返回：
    - out: 输出，形状为(N, M)
    - cache: 缓存数据 (x, w, b)
    """
    ###########################################################################
    # 待办：实现仿射层的前向传播。将结果存储在out中。你需要将输入重塑为行向量。    #
    ###########################################################################
    N = x.shape[0]
    x_flat = x.reshape(N,-1)
    out = x_flat@w + b
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    计算仿射层的反向传播。

    输入：
    - dout: 上游梯度，形状为(N, M)
    - cache: 缓存的元组：
      - x: 输入数据，形状为(N, d_1, ..., d_k)
      - w: 权重，形状为(D, M)
      - b: 偏置，形状为(M,)

    返回：
    - dx: 关于x的梯度，形状为(N, d1, ..., d_k)
    - dw: 关于w的梯度，形状为(D, M)
    - db: 关于b的梯度，形状为(M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # 待办：实现仿射层的反向传播。                                             #
    ###########################################################################
    N = x.shape[0]
    x_flat = x.reshape(N,-1)
    dx = dout @ w.T
    dx = dx.reshape(x.shape)
    dw = x_flat.T @ dout
    db = np.sum(dout,axis = 0)
    ###########################################################################
    #                             你的代码结束                                #
    ###########################################################################
    return dx, dw, db

def relu_affine_forward(x, w, b):
    #fc_cache里面存了x,w,b
    out,fc_cache = affine_forward(x,w,b)
    #relu_cache里面存了out,也就是算出来的中间值
    #由于我们在后续反向计算梯度的时候，需要用到中间值，所以得把这个中间值存下来
    out_relu,relu_cache= relu_forward(out)
    cache = (fc_cache,relu_cache)
    return out_relu,cache

def relu_dropout_affine_forward(x,w,b,dropout_param):
    out,fc_cache = affine_forward(x,w,b)
    out_relu,relu_cache= relu_forward(out)
    out_dropout,dropout_cache = dropout_forward(out_relu,dropout_param)
    cache = (fc_cache,relu_cache,dropout_cache)
    return out_dropout,cache

def relu_affine_backward(dout,cache):
    #反向传播的时候，先反向通过relu再affine_backward
    fc_cache, relu_cache = cache
    #relu_backward不需要知道x,w,b,只需要知道中间值，即relu_cache
    da = relu_backward(dout,relu_cache)
    #接下来的affine_backward才需要x,w,b以及中间值
    dx,dw,db = affine_backward(da,fc_cache)
    return dx,dw,db

def relu_dropout_affine_backward(dout,cache):
    fc_cache,relu_cache,dropout_cache = cache
    d_dropout = dropout_backward(dout,dropout_cache)
    d_relu = relu_backward(d_dropout,relu_cache)
    dx,dw,db = affine_backward(d_relu,fc_cache)
    return dx,dw,db

def batchnorm_relu_affine_forward(x, w, b,gamma,beta,bn_param):
    out,fc_cache = affine_forward(x,w,b)
    out_batch,batch_cache= batchnorm_forward(out,gamma,beta,bn_param)
    out_relu,relu_cache = relu_forward(out_batch)
    cache = (fc_cache,batch_cache,relu_cache)
    return out_relu,cache

def batch_relu_drop_affine_forward(x, w, b,gamma,beta,bn_param,dropout_param):
    out,fc_cache = affine_forward(x,w,b)
    out_batch,batch_cache= batchnorm_forward(out,gamma,beta,bn_param)
    out_relu,relu_cache = relu_forward(out_batch)
    out_dropout,dropout_cache = dropout_forward(out_relu,dropout_param)
    cache = (fc_cache,batch_cache,relu_cache,dropout_cache)
    return out_dropout,cache

def batchnorm_relu_affine_backward(dout,cache):
    fc_cache,batch_cache,relu_cache = cache
    dx_relu = relu_backward(dout,relu_cache)
    dx_batch,dgamma,dbeta = batchnorm_backward(dx_relu,batch_cache)
    dx,dw,db = affine_backward(dx_batch,fc_cache)
    return dx,dw,db,dgamma,dbeta

def batch_relu_drop_affine_backward(dout,cache):
    fc_cache,batch_cache,relu_cache,dropout_cache = cache
    dx_dropout = dropout_backward(dout,dropout_cache)
    dx_relu = relu_backward(dx_dropout,relu_cache)
    dx_batch,dgamma,dbeta = batchnorm_backward(dx_relu,batch_cache)
    dx,dw,db = affine_backward(dx_batch,fc_cache)
    return dx,dw,db,dgamma,dbeta

def layernorm_relu_affine_forward(x, w, b,gamma,beta,bn_param):
    out,fc_cache = affine_forward(x,w,b)
    out_batch,batch_cache= layernorm_forward(out,gamma,beta,bn_param)
    out_relu,relu_cache = relu_forward(out_batch)
    cache = (fc_cache,batch_cache,relu_cache)
    return out_relu,cache

def layer_relu_drop_affine_forward(x, w, b,gamma,beta,bn_param,dropout_param):
    out,fc_cache = affine_forward(x,w,b)
    out_batch,batch_cache= layernorm_forward(out,gamma,beta,bn_param)
    out_relu,relu_cache = relu_forward(out_batch)
    out_dropout,dropout_cache = dropout_forward(out_relu,dropout_param)
    cache = (fc_cache,batch_cache,relu_cache,dropout_cache)
    return out_dropout,cache

def layernorm_relu_affine_backward(dout,cache):
    fc_cache,batch_cache,relu_cache = cache
    dx_relu = relu_backward(dout,relu_cache)
    dx_batch,dgamma,dbeta = layernorm_backward(dx_relu,batch_cache)
    dx,dw,db = affine_backward(dx_batch,fc_cache)
    return dx,dw,db,dgamma,dbeta

def layer_relu_drop_affine_backward(dout,cache):
    fc_cache,batch_cache,relu_cache,dropout_cache = cache
    dx_dropout = dropout_backward(dout,dropout_cache)
    dx_relu = relu_backward(dx_dropout,relu_cache)
    dx_batch,dgamma,dbeta = layernorm_backward(dx_relu,batch_cache)
    dx,dw,db = affine_backward(dx_batch,fc_cache)
    return dx,dw,db,dgamma,dbeta

def relu_forward(x):
    """
    计算整流线性单元（ReLU）层的前向传播。

    输入：
    - x: 任意形状的输入

    返回：
    - out: 输出，与x形状相同
    - cache: 缓存x
    """
    out = None
    ###########################################################################
    # 待办：实现ReLU的前向传播。                                               #
    ###########################################################################
    out = np.maximum(x,0)
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    计算整流线性单元（ReLU）层的反向传播。

    输入：
    - dout: 上游梯度，任意形状
    - cache: 输入x，与dout形状相同

    返回：
    - dx: 关于x的梯度
    """
    dx, x = None, cache
    ###########################################################################
    # 待办：实现ReLU的反向传播。                                               #
    ###########################################################################
    dx = dout.copy()
    dx[x <= 0] = 0
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    return dx




def softmax_loss(x, y):
    """
    计算softmax分类的损失和梯度。

    输入：
    - x: 输入数据，形状为(N, C)，其中x[i, j]是第i个输入对第j类的得分
    - y: 标签向量，形状为(N,)，其中y[i]是x[i]的标签，且0 <= y[i] < C

    返回：
    - loss: 损失标量
    - dx: 损失关于x的梯度
    """
    #需要注意的是，这里的x其实是上一层计算出来的得分
    #可以这样理解：affine_forward+softmax_loss+affine_backward一起相当于softmax.py中的softmax_loss函数

    loss, dx = None, None
    x_max = np.max(x,axis=1,keepdims=True)
    x_exp = np.exp(x-x_max)
    x_exp_sum = np.sum(x_exp,axis=1,keepdims=True)
    x_prop = x_exp / x_exp_sum #得到了所有图片的属于哪个类别的可能性矩阵，(N,C)
    x_log = -np.log(x_prop)
    n = x.shape[0] #N
    one_hot = np.zeros(x.shape)
    row_index = np.arange(n)
    col_index = y    
    one_hot[row_index,col_index] = 1
    loss = x_log[row_index,col_index]
    dx = (x_prop-one_hot)/n
    loss = sum(loss) / n

    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """批归一化的前向传播。

    在训练期间，样本均值和（未校正的）样本方差从迷你批量统计中计算，并用于归一化输入数据。
    在训练期间，我们还保持每个特征的均值和方差的指数衰减运行平均值，这些平均值用于测试时的归一化。

    在每个时间步，我们使用基于动量参数的指数衰减来更新均值和方差的运行平均值：

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    注意，批归一化论文建议不同的测试时行为：他们使用大量训练图像计算每个特征的样本均值和方差，
    而不是使用运行平均值。在本实现中，我们选择使用运行平均值，因为它们不需要额外的估计步骤；
    torch7的批归一化实现也使用运行平均值。

    输入：
    - x: 形状为(N, D)的数据
    - gamma: 形状为(D,)的缩放参数
    - beta: 形状为(D,)的偏移参数
    - bn_param: 包含以下键的字典：
      - mode: 'train'或'test'；必需
      - eps: 数值稳定性的常数
      - momentum: 运行均值/方差的常数
      - running_mean: 形状为(D,)的特征运行均值数组
      - running_var: 形状为(D,)的特征运行方差数组

    返回：
    - out: 形状为(N, D)的输出
    - cache: 反向传播所需的中间值元组
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        ########################################################################
        # 实现批归一化的训练时前向传播。                                         #
        # 使用迷你批量统计计算均值和方差，使用这些统计量归一化输入数据，           #
        # 并使用gamma和beta对归一化数据进行缩放和偏移。                          #
        #                                                                      #
        # 应将输出存储在变量out中。反向传播所需的任何中间值应存储在cache变量中。   #
        #                                                                      #
        # 还应使用计算出的样本均值和方差以及动量变量来更新运行均值和运行方差，     #
        # 将结果存储在running_mean和running_var变量中。                         #
        #                                                                      #
        # 注意，尽管需要跟踪运行方差，但应基于标准差（方差的平方根）归一化数据！   #
        # 参考原始论文（https://arxiv.org/abs/1502.03167）可能会有帮助。         #
        ########################################################################
        #首先要计算每一列的均值和方差
        mean = np.mean(x, axis = 0)
        var = np.var(x, axis = 0)
        x_norm = (x - mean) / np.sqrt(var + eps)
        out = x_norm * gamma + beta
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var= momentum * running_var + (1 - momentum) * var
        #根据out = gamma(x - mean)/sqrt(var) + beta
        std = np.sqrt(var + eps)
        xmu = x - mean
        cache = (gamma, xmu, std,x_norm)
        

        ########################################################################
        #                           你的代码结束                                #
        ########################################################################
    elif mode == "test":
        ################################################################################
        # 实现批归一化的测试时前向传播。                                                 #
        # 使用运行均值和方差归一化输入数据，然后使用gamma和beta对归一化数据进行缩放和偏移。#
        # 将结果存储在out变量中。                                                      #
        ##############################################################################
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = x_norm * gamma + beta
        #######################################################################
        #                          你的代码结束                                 #
        #######################################################################
    else:
        raise ValueError('无效的批归一化前向模式 "%s"' % mode)

    # 将更新后的运行均值存储回bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """批归一化的反向传播。

    对于本实现，应在纸上画出批归一化的计算图，并通过中间节点反向传播梯度。

    输入：
    - dout: 上游导数，形状为(N, D)
    - cache: 来自batchnorm_forward的中间值变量。

    返回：
    - dx: 相对于输入x的梯度，形状为(N, D)
    - dgamma: 相对于缩放参数gamma的梯度，形状为(D,)
    - dbeta: 相对于偏移参数beta的梯度，形状为(D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # 实现批归一化的反向传播。将结果存储在dx、dgamma和dbeta变量中。              #
    # 参考原始论文（https://arxiv.org/abs/1502.03167）可能会有帮助。            #
    ###########################################################################
    N , D = dout.shape
    gamma,xmu,std,x_norm = cache
    dgamma =  np.sum(dout * x_norm, axis = 0) 
    dbeta = np.sum(dout, axis = 0) 
    dx_head = dout * gamma
    dxmu1 = dx_head / std
    distd = np.sum(dx_head * xmu, axis = 0)
    dstd = - 1 / (std ** 2) * distd
    dvar = 0.5 * dstd / std
    dsq = 1 / N * np.ones((N,D)) * dvar
    dxmu2 = 2 * xmu * dsq
    dxmu = dxmu1 + dxmu2
    dx1 = dxmu
    dmean = -np.sum(dxmu,axis = 0)
    dx2 = 1 / N * np.ones((N,D)) * dmean
    dx = dx1 + dx2
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """批归一化的替代反向传播。

    对于本实现，应在纸上计算批归一化反向传播的导数并尽可能简化。应能推导出反向传播的简单表达式。
    更多提示参见jupyter笔记本。

    注意：本实现应期望接收与batchnorm_backward相同的cache变量，但可能不会使用cache中的所有值。

    输入/输出：与batchnorm_backward相同
    """
    dx, dgamma, dbeta = None, None, None
    ############################################################################
    # 实现批归一化的反向传播。将结果存储在dx、dgamma和dbeta变量中。               #
    #                                                                          #
    # 在计算相对于中心化输入的梯度后，应能在单个语句中计算相对于输入的梯度；       #
    # 我们的实现适合在一行80个字符内。                                          # 
    ###########################################################################
    N , D = dout.shape
    gamma,xmu,std,x_norm = cache
    dgamma =  np.sum(dout * x_norm, axis = 0) 
    dbeta = np.sum(dout, axis = 0) 
    dx = gamma / (N * std) *( N * dout - np.sum(dout, axis = 0) - x_norm * np.sum(x_norm * dout, axis = 0))
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """层归一化的前向传播。

    在训练和测试时，输入数据都按每个数据点进行归一化，然后使用与批归一化相同的gamma和beta参数进行缩放和偏移。

    注意，与批归一化不同，层归一化在训练和测试时的行为是相同的，不需要跟踪任何运行平均值。

    输入：
    - x: 形状为(N, D)的数据
    - gamma: 形状为(D,)的缩放参数
    - beta: 形状为(D,)的偏移参数
    - ln_param: 包含以下键的字典：
        - eps: 数值稳定性的常数

    返回：
    - out: 形状为(N, D)的输出
    - cache: 反向传播所需的中间值元组
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ##############################################################################
    # 实现层归一化的训练时前向传播。                                               #
    # 归一化输入数据，并使用gamma和beta对归一化数据进行缩放和偏移。                 #
    # 提示：这可以通过稍微修改批归一化的训练时实现，并插入一两行精心设计的代码来完成。#
    # 特别是，能否想到任何矩阵变换，可以使你复制批归一化代码并几乎不做修改？         #
    #############################################################################
    
    mean = np.mean(x, axis = 1,keepdims = True) #记得要keepdims,不然就变成行向量了
    var = np.var(x, axis = 1,keepdims = True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    out = x_norm * gamma + beta
    #根据out = gamma(x - mean)/sqrt(var) + beta
    std = np.sqrt(var + eps)
    xmu = x - mean
    cache = (gamma, xmu, std,x_norm)
        

    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """层归一化的反向传播。

    对于本实现，可以在很大程度上依赖已经为批归一化所做的工作。

    输入：
    - dout: 上游导数，形状为(N, D)
    - cache: 来自layernorm_forward的中间值变量。

    返回：
    - dx: 相对于输入x的梯度，形状为(N, D)
    - dgamma: 相对于缩放参数gamma的梯度，形状为(D,)
    - dbeta: 相对于偏移参数beta的梯度，形状为(D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # 实现层归一化的反向传播。                                                  #
    #                                                                         #
    # 提示：这可以通过稍微修改批归一化的训练时实现来完成。前向传播的提示仍然适用！ #
    ###########################################################################
    #注意，ln虽然是横向归一化，但是gamma,beta仍然是作用于特征的，也就是依然是纵向的
    N , D = dout.shape
    gamma,xmu,std,x_norm = cache
    dgamma =  np.sum(dout * x_norm, axis = 0) 
    dbeta = np.sum(dout, axis = 0)
    dx_head = dout * gamma
    dxmu1 = dx_head / std
    distd = np.sum(dx_head * xmu, axis = 1,keepdims = True)#(N,1)
    dstd = - 1 / (std ** 2) * distd
    dvar = 0.5 * dstd / std #(N,1)
    dsq = 1 / D * np.ones((N,D)) * dvar
    dxmu2 = 2 * xmu * dsq
    dxmu = dxmu1 + dxmu2
    dx1 = dxmu
    dmean = -np.sum(dxmu,axis = 1,keepdims = True)
    dx2 = 1 / D * np.ones((N,D)) * dmean
    dx = dx1 + dx2
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    return dx, dgamma, dbeta

def dropout_forward(x, dropout_param):
    """倒置丢弃法（inverted dropout）的前向传播。

    注意这与标准丢弃法不同。这里，p是保留神经元输出的概率，而非丢弃神经元输出的概率。
    更多细节参见http://cs231n.github.io/neural-networks-2/#reg。

    输入：
    - x: 任意形状的输入数据
    - dropout_param: 包含以下键的字典：
      - p: 丢弃参数。我们以概率p保留每个神经元的输出。
      - mode: 'test'或'train'。若为训练模式，则执行丢弃；若为测试模式，则直接返回输入。
      - seed: 随机数生成器的种子。传入种子可使函数具有确定性，这在梯度检查中需要，但在实际网络中不需要。

    输出：
    - out: 与x形状相同的数组。
    - cache: 元组(dropout_param, mask)。训练模式下，mask是用于与输入相乘的丢弃掩码；测试模式下，mask为None。
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])  # 设置随机种子以保证确定性

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # 实现训练阶段的倒置丢弃法前向传播。将丢弃掩码存储在mask变量中。         #
        #######################################################################
        mask = np.random.binomial(1, p, size=x.shape) / p
        out = x * mask
        #######################################################################
        #                           你的代码结束                               #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # 实现测试阶段的倒置丢弃法前向传播。                                    #
        #######################################################################
        out = x
        #######################################################################
        #                            你的代码结束                              #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)  # 确保输出数据类型与输入一致

    return out, cache


def dropout_backward(dout, cache):
    """倒置丢弃法的反向传播。

    输入：
    - dout: 上游导数，任意形状
    - cache: 来自dropout_forward的(dropout_param, mask)
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # 实现训练阶段的倒置丢弃法反向传播。                                    #
        #######################################################################
        p = dropout_param["p"]
        dx = dout.copy()
        dx[mask == 0] = 0
        dx[mask != 0] /= p
        #######################################################################
        #                          你的代码结束                                 #
        #######################################################################
    elif mode == "test":
        dx = dout  # 测试模式下，梯度直接传递
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """卷积层前向传播的朴素实现。

    输入包含N个数据点，每个数据点有C个通道、高度H和宽度W。我们使用F个不同的滤波器对每个输入进行卷积，
    每个滤波器覆盖所有C个通道，高度为HH，宽度为WW。

    输入：
    - x: 输入数据，形状为(N, C, H, W)
    - w: 滤波器权重，形状为(F, C, HH, WW)
    - b: 偏置，形状为(F,)
    - conv_param: 包含以下键的字典：
      - 'stride': 水平和垂直方向上相邻感受野之间的像素数（步长）。
      - 'pad': 用于对输入进行零填充的像素数。

    填充时，应在输入的高度和宽度轴上对称地放置'pad'个零（即两侧各放pad个）。注意不要直接修改原始输入x。

    返回：
    - out: 输出数据，形状为(N, F, H', W')，其中H'和W'由下式计算：
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # 实现卷积前向传播。提示：可以使用np.pad函数进行填充。                       #
    ###########################################################################
    #先填充
    N,_,H,W = x.shape
    F,_,HH,WW = w.shape
    stride = conv_param["stride"]
    pad = conv_param["pad"]
    pad_width = (
    (0, 0),          # N 维度：前后都不填
    (0, 0),          # C 维度：前后都不填
    (pad, pad), # H 维度：上下各填 padding 个
    (pad, pad)  # W 维度：左右各填 padding 个
)
    x_pad = np.pad(x, pad_width, mode='constant', constant_values=0)
    H_head = 1 + (H + 2 * pad - HH) // stride
    W_head = 1 + (W + 2 * pad - WW) // stride
    out = np.zeros((N, F, H_head, W_head))
    for i in range(N):
        for j in range(F):
            for m in range(H_head):
                for n in range(W_head):
                    out[i,j,m,n] = np.sum(x_pad[i, : ,m*stride : m*stride + HH,n*stride : n*stride + WW] * w[j,:,:,:]) + b[j]
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """卷积层反向传播的朴素实现。

    输入：
    - dout: 上游导数。
    - cache: 来自conv_forward_naive的(x, w, b, conv_param)元组

    返回：
    - dx: 相对于x的梯度
    - dw: 相对于w的梯度
    - db: 相对于b的梯度
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # 实现卷积反向传播。                                                       #
    ###########################################################################
    x, w, b,conv_param = cache
    N,C,H,W = x.shape
    F,_,HH,WW = w.shape
    stride = conv_param["stride"]
    pad = conv_param["pad"]
    pad_width = (
    (0, 0),          # N 维度：前后都不填
    (0, 0),          # C 维度：前后都不填
    (pad, pad), # H 维度：上下各填 padding 个
    (pad, pad)  # W 维度：左右各填 padding 个
)
    x_pad = np.pad(x, pad_width, mode='constant', constant_values=0)
    H_head = 1 + (H + 2 * pad - HH) // stride
    W_head = 1 + (W + 2 * pad - WW) // stride
    dx_pad = np.zeros(x_pad.shape)
    dw = np.zeros((F,C,HH,WW))
    db = np.zeros((F,))
    for i in range(N):
        for j in range(F):
            for m in range(H_head):
                for n in range(W_head):
                    x_slice = x_pad[i, : ,m*stride : m*stride + HH,n*stride : n*stride + WW] 
                    #就是把上面前向传播的过程反过来
                    db[j] += dout[i,j,m,n]
                    dw[j,:,:,:] += x_slice * dout[i,j,m,n]
                    dx_pad[i, : ,m*stride : m*stride + HH,n*stride : n*stride + WW] += w[j] * dout[i,j,m,n]
    dx = dx_pad[:, :, pad:-pad, pad:-pad]
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """最大池化层前向传播的朴素实现。

    输入：
    - x: 输入数据，形状为(N, C, H, W)
    - pool_param: 包含以下键的字典：
      - 'pool_height': 每个池化区域的高度
      - 'pool_width': 每个池化区域的宽度
      - 'stride': 相邻池化区域之间的距离

    这里不需要填充，例如可假设：
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    返回：
    - out: 输出数据，形状为(N, C, H', W')，其中H'和W'由下式计算：
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # 实现最大池化前向传播。                                                   #
    ###########################################################################
    N,C,H,W = x.shape
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]
    H_head = 1 + (H - pool_height) // stride
    W_head = 1 + (W - pool_width) // stride
    out = np.zeros((N,C,H_head,W_head))
    for i in range(N):
        for j in range(C):
            for m in range(H_head):
                for n in range(W_head):
                    x_slice = x[i, j ,m*stride : m*stride + pool_height,n*stride : n*stride + pool_width]
                    max_values = np.max(x_slice, axis=(0, 1))
                    out[i,j,m,n] = max_values
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """最大池化层反向传播的朴素实现。

    输入：
    - dout: 上游导数
    - cache: 来自前向传播的(x, pool_param)元组

    返回：
    - dx: 相对于x的梯度
    """
    dx = None
    ###########################################################################
    # 实现最大池化反向传播。                                                   #
    ###########################################################################
    # 要点就是要找到x里面每个窗口的最大值，只有那个值传递了梯度，其他的地方都是0
    x,pool_param = cache
    dx = np.zeros(x.shape)
    N,C,H_head,W_head = dout.shape
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]
    for i in range(N):
        for j in range(C):
            for m in range(H_head):
                for n in range(W_head):
                    x_slice = x[i, j ,m*stride : m*stride + pool_height,n*stride : n*stride + pool_width]
                    max_value = np.max(x_slice)
                    mask = (x_slice == max_value) #这会创建一个和x_slice一样尺寸的矩阵，在和max_value一样的位置为true,否则为false
                    dx[i, j ,m*stride : m*stride + pool_height,n*stride : n*stride + pool_width] = mask * dout[i,j,m,n]

    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """空间批归一化的前向传播。

    输入：
    - x: 输入数据，形状为(N, C, H, W)
    - gamma: 缩放参数，形状为(C,)
    - beta: 偏移参数，形状为(C,)
    - bn_param: 包含以下键的字典：
      - mode: 'train'或'test'；必需
      - eps: 数值稳定性常数
      - momentum: 运行均值/方差的常数。momentum=0表示每次完全丢弃旧信息，
        而momentum=1表示从不纳入新信息。默认momentum=0.9在大多数情况下适用。
      - running_mean: 形状为(D,)的特征运行均值数组
      - running_var: 形状为(D,)的特征运行方差数组

    返回：
    - out: 输出数据，形状为(N, C, H, W)
    - cache: 反向传播所需的值
    """
    out, cache = None, None

    ###########################################################################
    # 实现空间批归一化的前向传播。                                              #
    #                                                                         #
    # 提示：可通过调用上面实现的标准批归一化函数来实现空间批归一化。              #
    # 实现应该非常简短；我们的实现不到5行。                                     #
    ###########################################################################
    #可以通过把维度重塑的方式对之前的代码进行复用
    #把（N,C,H,W)重塑为（N*H*W，C）
    N, C, H, W = x.shape
    x_transposed = x.transpose(0, 2, 3, 1) #（N,H,W,C）
    x_flat = x_transposed.reshape(-1, C)
    out,cache = batchnorm_forward(x_flat,gamma,beta,bn_param)
    out = out.reshape(N, H, W, C)
    out = out.transpose(0,3,1,2)
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """空间批归一化的反向传播。

    输入：
    - dout: 上游导数，形状为(N, C, H, W)
    - cache: 前向传播中的值

    返回：
    - dx: 相对于输入的梯度，形状为(N, C, H, W)
    - dgamma: 相对于缩放参数的梯度，形状为(C,)
    - dbeta: 相对于偏移参数的梯度，形状为(C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # 实现空间批归一化的反向传播。                                              #
    #                                                                         #
    # 提示：可通过调用上面实现的标准批归一化函数来实现空间批归一化。              #
    # 实现应该非常简短；我们的实现不到5行。                                     #
    ###########################################################################
    N, C, H, W = dout.shape
    dout_transposed = dout.transpose(0, 2, 3, 1) #（N,H,W,C)
    dout_flat = dout_transposed.reshape(-1, C)
    dx,dgamma,dbeta = batchnorm_backward(dout_flat,cache)
    dx = dx.reshape(N, H, W, C)
    dx = dx.transpose(0,3,1,2)
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """空间组归一化的前向传播。
    
    与层归一化不同，组归一化将数据中的每个样本分成G个连续的部分，然后独立地对每个部分进行归一化。
    然后对数据应用逐特征的偏移和缩放，方式与批归一化和层归一化相同。

    输入：
    - x: 输入数据，形状为(N, C, H, W)
    - gamma: 缩放参数，形状为(1, C, 1, 1)
    - beta: 偏移参数，形状为(1, C, 1, 1)
    - G: 要划分的组数，必须是C的约数
    - gn_param: 包含以下键的字典：
      - eps: 数值稳定性常数

    返回：
    - out: 输出数据，形状为(N, C, H, W)
    - cache: 反向传播所需的值
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # 实现空间组归一化的前向传播。                                              #
    # 这与层归一化的实现极其相似。                                              #
    # 具体来说，思考如何转换矩阵，使得大部分代码可复用训练时的批归一化和层归一化！ #
    ###########################################################################
    N, C, H, W = x.shape
    #所谓分组就是把c个通道分组
    #最大的不同在于要计算N*G个均值
    x_flat = x.reshape(N*G,-1)
    dummy_gamma = np.ones(x_flat.shape[1])
    dummy_beta = np.zeros(x_flat.shape[1])
    ln_xnorm,ln_cache = layernorm_forward(x_flat,dummy_gamma,dummy_beta,gn_param)
    x_norm = ln_xnorm.reshape(N, C, H, W)
    out = x_norm * gamma + beta
    cache = (ln_cache,G,x_norm,gamma)
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """空间组归一化的反向传播。

    输入：
    - dout: 上游导数，形状为(N, C, H, W)
    - cache: 前向传播中的值

    返回：
    - dx: 相对于输入的梯度，形状为(N, C, H, W)
    - dgamma: 相对于缩放参数的梯度，形状为(1, C, 1, 1)
    - dbeta: 相对于偏移参数的梯度，形状为(1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # 实现空间组归一化的反向传播。                                              #
    # 这与层归一化的实现极其相似。                                              #
    ###########################################################################
    #最要注意的点在于我们把out = x_norm * gamma + beta放到函数外面来计算了
    #先在外面把dbeta,dgamma处理掉
    N, C, H, W = dout.shape
    ln_cache,G,x_norm,gamma= cache #这里的x_norm是(N,C,H,W)
    dbeta = np.sum(dout,axis = (0,2,3),keepdims = True)
    dgamma = np.sum(dout * x_norm,axis = (0,2,3) ,keepdims = True)
    #由于out = x_norm * gamma + beta是在函数之外写的，所以带入到函数里面，dx_norm才是要带入到函数里面的dout
    #所以要先对包含了x的x_norm进行反向传播求导，注意，这里不需要求和了，因为 维度是一样的
    dx_norm = dout * gamma
    dx_norm_flatten = dx_norm.reshape(N*G,-1)
    dx_flatten,_,_ = layernorm_backward(dx_norm_flatten,ln_cache)
    dx = dx_flatten.reshape(N,C,H,W)
    ###########################################################################
    #                             你的代码结束                                 #
    ###########################################################################
    return dx, dgamma, dbeta
