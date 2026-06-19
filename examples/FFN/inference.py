# YO
# Welcome to the Libary Introduction
# This example will display how the library training and inference process would look like.
# Here, we will try to approximate the function f(x, y) = e^(-x^2 - y^2)

# This is the source code for the inference process
# I hope the whole pipeline is self-explainable enough
# Note that we should explicitly call deactivate_backprop 
# so that it doesn't waste time saving an extra gradient array

import random
import math
import time
import lktorch as lk
import os


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_PATH = os.path.join(CURRENT_DIR, "training_data", "model_weight.cache")

start_checkpoint = time.perf_counter()

private_test = []
for i in range(-200, 201):
    for j in range(-200, 201):
        private_test.append([i / 10, j / 10, math.exp(- (i / 10)**2 - (j / 10) ** 2)])

random.shuffle(private_test)

private_test_tensor = lk.Tensor(private_test)
input_data = private_test_tensor.Slice([0], [1])
output_data = private_test_tensor.Slice([2], [2])

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

lk.deactivate_backprop()


def calculate_loss(my_nn, input, output, LossFunction):
    tenso = my_nn(input)
    tenso = LossFunction(tenso, output)
    return lk.MeanAll(tenso).accessA([])

start_checkpoint = time.perf_counter()
print("No training:")
my_output = model(input_data)
mse_loss = lk.MeanAll(lk.MSELoss(my_output, output_data))
mae_loss = lk.MeanAll(lk.MAELoss(my_output, output_data))
rmse_loss = lk.MeanAll(lk.RMSELoss(my_output, output_data))
print("MSE Loss: ", mse_loss)
print("MAE Loss: ", mae_loss)
print("RMSE Loss: ", rmse_loss)

end_checkpoint = time.perf_counter()
print("Time elapsed in inference: ", end_checkpoint - start_checkpoint, " second(s)")


model.load_state_dict(lk.ReadFile(WEIGHT_PATH))


start_checkpoint = time.perf_counter()
print("Yes training:")
my_output = model(input_data)
mse_loss = lk.MeanAll(lk.MSELoss(my_output, output_data))
mae_loss = lk.MeanAll(lk.MAELoss(my_output, output_data))
rmse_loss = lk.MeanAll(lk.RMSELoss(my_output, output_data))
print("MSE Loss: ", mse_loss)
print("MAE Loss: ", mae_loss)
print("RMSE Loss: ", rmse_loss)
end_checkpoint = time.perf_counter()
print("Time elapsed in inference: ", end_checkpoint - start_checkpoint, " second(s)")
