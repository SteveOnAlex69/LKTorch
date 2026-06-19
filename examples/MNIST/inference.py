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
DATA_PATH = os.path.join(CURRENT_DIR, "training_data", "test.csv")
RESULT_PATH = os.path.join(CURRENT_DIR, "result", "result.csv")



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
label = []

for i in range(row_count):
    input_row = input.loc[i].to_numpy()
    input_dataset.append(input_row)

input_tensor = lk.Tensor(input_dataset) / 255


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

end_checkpoint = time.perf_counter()
print("Time elapsed in loading data: ", end_checkpoint - start_checkpoint, " second(s)")
lk.deactivate_backprop()

model.load_state_dict(lk.ReadFile(WEIGHT_PATH))
start_checkpoint = time.perf_counter()
tenso = model(input_tensor)
end_checkpoint = time.perf_counter()

label = get_label(tenso)


output_d = {"ImageId": [i for i in range(1, row_count + 1)], "Label": label}
data_frame = pd.DataFrame(data = output_d)
print(data_frame)
data_frame.to_csv(RESULT_PATH, index=False)