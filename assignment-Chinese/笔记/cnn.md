![alt text](3cd6c443191357dc75f10e08b7a4af08.png)
![alt text](9e115460bddafff1769428289e00b9e6.png)
![alt text](0d285d9e2a2e7f4aff99b522dd265e85.png)
![alt text](e91ac3988546b15285c73d4e8b982366.png)
关于参数共享：
![alt text](05a1053fa7e739d66d9ac1536d33cc76.png)
比如上图的例子，我们使用5个滤波器，于是图片在经过第一层卷积计算之后变成了5层，我们算出来的这5层如果还用一个全连接的大矩阵来乘，那么要计算的参数量就太多了。

我们可以这样理解，每一层相仿于是某个特征在图上的探测结果，那么这个特征出现的位置实际上是无所谓的，我们更关注的是他有没有出现，换言之，我们对于同一层可以使用一个参数
但是也要注意，参数共享并不是总是适用的 

#### 归一化
![alt text](bfde25675b002d4070af4a210e4c8d81.png)
#### dropout
![alt text](5b90a8a5e7c34b203abfbd65d3f8831a.png)
![alt text](51990d27d2b017f213f767ba747087ad.png)

#### ResNet(可以有助于训练更深层次的网络)
![alt text](670fe8001f39bebde1e7e214a55bca7f.png)
![alt text](75fbe2ab64dbf7b59759380271283010.png)

#### 数据增强
![alt text](a48be0a2651a09afefaa2ad3554eca9d.png)
![alt text](c254b0eb1c86234df4eaf47f2f92191b.png)