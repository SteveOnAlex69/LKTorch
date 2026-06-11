import lktorch as lk

tenso1 = lk.Tensor([3, 6])
tenso2 = lk.Tensor([3, 6])

tenso = tenso1 + tenso2

li = lk.LinearLayer(6, 3)

tenso = li.forward(tenso)

tenso1 = lk.Sigmoid(tenso)
tenso2 = tenso
tenso1 = lk.SumAll(tenso1)
tenso2 = lk.SumAll(tenso2)

print(tenso1.accessA([]), tenso2.accessA([]))

print("Forward Passs")