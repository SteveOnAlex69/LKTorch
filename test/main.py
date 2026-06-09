import lktorch as lk

tenso1 = lk.Tensor([3, 6])
tenso2 = lk.Tensor([3, 6])

tenso = tenso1 + tenso2

li = lk.LinearLayer(6, 3)

tenso = li.forward(tenso)

tenso = tenso.Reshape([9])

print("Forward Passs")