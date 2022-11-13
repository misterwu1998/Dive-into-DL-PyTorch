# 第0行包含了最大元素，但是第0行之和比其它行的小
# t=torch.tensor(
#     data=[[ 1.3398,  0.2663, -100.2686,  0.2450],
#         [-0.7401, -0.8805, -0.3402, -1.1936],
#         [ 0.4907, -1.3948, -1.0691, -0.3132],
#         [-1.6092,  0.5419, -0.2993,  0.3195]],
#     requires_grad=False)
# print(torch.argmax(t)) # “tensor(0)”
# 说明即使不指定dim，torch.argmax()也是直接找最大元素而不是其它维度求和后找最大“行”

# a=torch.tensor([1,2,3])
# b=torch.tensor([2,2,2])
# e=(a==b)
# print(e) #tensor([False,  True, False])
# f=e.float()
# print(f) #tensor([0., 1., 0.])

# x=torch.tensor([1.0])
# i=x.item()
# print(i) #1.0
# print(x.data)#tensor([1.])

# 章节3.7
# 如何使用torch.nn.CrossEntropyLoss()
# loss=torch.nn.CrossEntropyLoss()
# input=torch.randn(size=(3,5),requires_grad=True)
# target=torch.empty(size=(3,),dtype=torch.long).random_(5)
# output=loss(input,target)
# output.backward()

# Tensor对象.view()不会改变任意一对元素之间的相对顺序。
# t=torch.tensor([
#     [
#         [1,2],
#         [3,4]
#     ],
#     [
#         [5,6],
#         [7,8]
#     ]
# ],dtype=torch.uint8)
# print(t)
# t=t.view(-1,t.size()[-1])
# print(t)

# def returnTuple():
#     return (1,2,3)
# a,b,c=returnTuple()
# print(a)

# def returnStr():
#     return '''
#         This is a string.
#     '''
# print(returnStr())

print('------------------------end------------------------')