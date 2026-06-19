# YO
# Welcome to the Libary Introduction
# This example will display how the library training and inference process would look like.
# Here, we will try to do mnist with feed forward network

# This is the source code for the training process
# I hope the whole pipeline is self-explainable enough

import pandas as pd
import time
import lktorch as lk
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_PATH = os.path.join(CURRENT_DIR, "training_data", "model_weight.cache")
DATA_PATH = os.path.join(CURRENT_DIR, "training_data", "train.csv")

model = lk.Sequential([
    lk.LinearLayer(784, 32),
    lk.reLU_Layer(),
    lk.LinearLayer(32, 16),
    lk.reLU_Layer(),
    lk.LinearLayer(16, 10),
    lk.Sigmoid_Layer(),
    lk.SoftMax_Layer()
])

start_checkpoint = time.perf_counter()
input = pd.read_csv(DATA_PATH)
row_count, column_count = input.shape

input_dataset = []
output_dataset = []
for i in range(row_count):
    input_row = input.loc[i].to_numpy()

    current_output = input_row[0]
    current_row = [0] * 10
    current_row[current_output] = 1
    output_dataset.append(current_row)
    input_dataset.append(input_row[1:])

input_tensor = lk.Tensor(input_dataset) / 255
output_tensor = lk.Tensor(output_dataset)

end_checkpoint = time.perf_counter()
print("Time elapsed in loading data: ", end_checkpoint - start_checkpoint, " second(s)")

optimizer = lk.Adam(2, 0.9, 0.999)
optimizer.add_parameter(model.get_parameters())

def calculate_loss(model, input_data, output_data, LossFunction):
    tenso = model(input_data)
    tenso = LossFunction(tenso, output_data)
    return lk.Mean(tenso)

start_checkpoint = time.perf_counter()
for epoch in range(1, 11):
    BATCH = 42
    l = 0
    while (l < len(input_dataset)):
        optimizer.zero_gradient()

        r = min(l + BATCH, len(input_dataset)) - 1

        input_batch = input_tensor.Slice([l, 0], [r, 783])
        tenso = model(input_batch)
        tenso = lk.Mean(lk.MSELoss(tenso, output_tensor.Slice([l, 0], [r, 9])))

        tenso.backward()
        optimizer.step()
        l += BATCH


    print("epoch:", epoch)


end_checkpoint = time.perf_counter()
print("Time elapsed in training: ", end_checkpoint - start_checkpoint, " second(s)")

lk.WriteFile(model.get_state_dict(), WEIGHT_PATH)
print("File saved successfully!")
