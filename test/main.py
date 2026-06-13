import math
import time
import lktorch as lk

arr1 = []
arr2 = []
for i in range(-20, 21):
    for j in range(-20, 21):
        arr1.append(i / 2)
        arr1.append(j / 2)
        arr2.append(math.exp(- (i / 2)**2 - (j / 2) ** 2))

input_data = lk.Tensor([len(arr2), 2], arr1)
output_data = lk.Tensor([len(arr2), 1], arr2)

model = lk.Sequential([
    lk.LinearLayer(2, 16),
    lk.reLU_Layer(),
    lk.LinearLayer(16, 16),
    lk.reLU_Layer(),
    lk.LinearLayer(16, 1),
    lk.Sigmoid_Layer(),
])

def calculate_loss(my_nn, input, output):
    tenso = my_nn(input)
    tenso = lk.MSELoss(tenso, output)
    return lk.MeanAll(tenso).accessA([])

cur = lk.ReadFile("E:/CollegePrj/Game/LKTorch/test/tmp.txt")
model.load_state_dict(cur)
print("Loss before: ", f"{calculate_loss(model, input_data, output_data):.10f}")

start_checkpoint = time.perf_counter()

optimizer = lk.Adam(0.001, 0.99, 0.99)
optimizer.add_parameter(model.get_parameters())

for epoch in range(1, 1001):
    batch = 101
    l = 0
    while (l < len(arr2)):
        r = min(l + batch - 1, len(arr2) - 1)
        input_batch = input_data.Slice([l, 0], [r, 1])
        output_batch = output_data.Slice([l, 0], [r, 0])

        tenso = model(input_batch)
        tenso = lk.RMSELoss(tenso, output_batch)
        tenso = lk.MeanAll(tenso)
        tenso.backward()

        l += batch
        optimizer.step()


print("Loss after: ", f"{calculate_loss(model, input_data, output_data):.10f}")


end_checkpoint = time.perf_counter()
print("Time elapsed: ", end_checkpoint - start_checkpoint)