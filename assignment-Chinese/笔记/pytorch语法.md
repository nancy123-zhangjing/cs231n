```python
a = torch.rand(2, 4) * 2 - 1
print('Common functions:')
print(torch.abs(a))
print(torch.ceil(a)) #向上取整
print(torch.floor(a)) #向下取整
print(torch.clamp(a, -0.5, 0.5)) #卡一个区间，超过区间的就取区间值

# Reshape
a = torch.arange(4.) # 会让生成的是浮点数
a_reshaped = torch.reshape(a, (2, 2))
b = torch.tensor([[0, 1], [2, 3]])
b_reshaped =torch.reshape(b, (-1,)) #reshape成一行
```
自动计算梯度