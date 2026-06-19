# YO
# Welcome to the Libary Introduction
# This example will display how the library training and inference process would look like.
# Here, we will try to approximate the function f(x, y) = e^(-x^2 - y^2)
# This is FFN (Feed Forward Network)!
# We are given a function, and we approximate it

# This is the source code for the training process
# I hope the whole pipeline is self-explainable enough

import random
import math
import time
import lktorch as lk
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_PATH = os.path.join(CURRENT_DIR, "training_data", "model_weight.cache")

start_checkpoint = time.perf_counter()

public_test = []
for i in range(-20, 21):
    for j in range(-20, 21):
        public_test.append([i / 2, j / 2, math.exp(- (i / 2)**2 - (j / 2) ** 2)])

random.shuffle(public_test)

public_test_tensor = lk.Tensor(public_test)
input_data = public_test_tensor.Slice([0], [1])
output_data = public_test_tensor.Slice([2], [2])

end_checkpoint = time.perf_counter()
print("Time elapsed in data fetching: ", end_checkpoint - start_checkpoint, " second(s)")

model = lk.Sequential([
    lk.LinearLayer(2, 16),
    lk.reLU_Layer(),
    lk.LinearLayer(16, 16),
    lk.reLU_Layer(),
    lk.LinearLayer(16, 1),
    lk.Sigmoid_Layer(),
])

start_checkpoint = time.perf_counter()

optimizer = lk.Adam(0.001, 0.99, 0.99)
optimizer.add_parameter(model.get_parameters())
for epoch in range(1, 1001):
    batch = 101
    l = 0
    while (l < len(public_test)):
        r = min(l + batch - 1, len(public_test) - 1)
        input_batch = input_data.Slice([l, 0], [r, 1])
        output_batch = output_data.Slice([l, 0], [r, 0])

        tenso = model(input_batch)
        tenso = lk.RMSELoss(tenso, output_batch)
        tenso = lk.MeanAll(tenso)
        tenso.backward()

        l += batch
        optimizer.step()


end_checkpoint = time.perf_counter()
print("Time elapsed in training: ", end_checkpoint - start_checkpoint, " second(s)")

lk.WriteFile(model.get_state_dict(), WEIGHT_PATH)
print("File saved successfully!")