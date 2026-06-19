# YO
# Welcome to the Libary Introduction
# This example will display how the library training and inference process would look like.
# Here, we will try to do mnist with feed forward network

# This is the source code for the inference process. 
# We will measure how it perform on the dataset
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
label = []

for i in range(row_count):
    input_row = input.loc[i].to_numpy()

    label.append(input_row[0])
    current_output = input_row[0]
    current_row = [0] * 10
    current_row[current_output] = 1
    output_dataset.append(current_row)
    input_dataset.append(input_row[1:])

input_tensor = lk.Tensor(input_dataset) / 255
output_tensor = lk.Tensor(output_dataset)

def get_label(tensor):
    label = []
    sz = tensor.dimension()
    for i in range(sz[0]):
        cur = -1
        ma = -1
        for j in range(0, 10):
            current_w = tensor.accessA([i, j])
            if (current_w > ma):
                ma = current_w
                cur = j
        label.append(cur)
    return label

def calculate_fault(tensor, label):
    correct = 0
    my_label = get_label(tensor)
    for i in range(len(label)):
        if (my_label[i] == label[i]):
            correct += 1
    return correct / len(label)

end_checkpoint = time.perf_counter()
print("Time elapsed in loading data: ", end_checkpoint - start_checkpoint, " second(s)")

lk.deactivate_backprop()

start_checkpoint = time.perf_counter()
print("No training")
tenso = model(input_tensor)
end_checkpoint = time.perf_counter()


print("Time elapsed in inference: ", end_checkpoint - start_checkpoint, " second(s)")
print("Accuracy: ", calculate_fault(tenso, label))

model.load_state_dict(lk.ReadFile(WEIGHT_PATH))
start_checkpoint = time.perf_counter()
print("Yes training")
tenso = model(input_tensor)
end_checkpoint = time.perf_counter()
print("Accuracy: ", calculate_fault(tenso, label))
print("Time elapsed in inference: ", end_checkpoint - start_checkpoint, " second(s)")

print("Sanity check: ")
print("Accuracy of output dataset: ", calculate_fault(output_tensor, label))