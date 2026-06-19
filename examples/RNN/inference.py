# YO
# Welcome to the Libary Introduction
# This example will display how the library training and inference process would look like.
# Here, we will try to approximate the function f(t) = sin(t)
# But there is a catch: We are not given the entire function at once. 
# We are only given a data point, f(t), and we have to predict f(t + PI / 50)
# Then we will feed the data point again
# This is RNN (Recurrent Neural Network)

# This is the source code for the training process
# Basically, we have (public state, hidden state)
# We take the public state to calculate loss, and use the (correct output, hidden state) to feed again
# Basically, we are forcing the public state to be correct, and the hidden state to also setup nicely for the public state

import random
import math
import time
import lktorch as lk
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_PATH = os.path.join(CURRENT_DIR, "training_data", "model_weight.cache")

start_checkpoint = time.perf_counter()

private_test = []
for i in range(0, 200):
    current_test = []
    for j in range(0, 100):
        current_test.append(math.sin(math.pi * (i + j * 0.5) / 50))
    private_test.append(current_test)

# Architecture: 1 explicit state + 3 hidden state
# The explicit state will be at slot 0, hidden state will be at slot [1, 2, 3]
model = lk.Sequential([
    lk.LinearLayer(4, 8),
    lk.Tanh_Layer(),
    lk.LinearLayer(8, 8),
    lk.Tanh_Layer(),
    lk.LinearLayer(8, 4),
    lk.Tanh_Layer(),
])

lk.deactivate_backprop()

end_checkpoint = time.perf_counter()
print("Time elapsed in data fetching: ", end_checkpoint - start_checkpoint, " second(s)")


def join_info(public_state, hidden_state):
    sz = len(public_state.dimension())
    perm = [sz-1]
    for i in range(sz-1): 
        perm.append(i)
    public_state = public_state.PermuteDimension(perm)
    hidden_state = hidden_state.PermuteDimension(perm)
    state = lk.Merge(public_state, hidden_state)

    invperm = []
    for i in range(sz - 1): 
        invperm.append(i + 1)
    invperm.append(0)
    return state.PermuteDimension(invperm)

def extract_public(tensor):
    return tensor.Slice([0], [0])

def extract_hidden(tensor):
    return tensor.Slice([1], [3])

def calculate_loss(model, LossFunction):
    input_data = lk.Tensor(private_test[0])
    dim = input_data.dimension()
    input_data = input_data.Reshape(dim + [1])
    input_data = join_info(input_data, lk.Zeros(dim + [3]))

    cur_tensor = input_data
    cum_loss = lk.Zeros([])
    for it in range(1, len(private_test)):
        cur_tensor = model(cur_tensor)
        public_state = extract_public(cur_tensor)
        hidden_state = extract_hidden(cur_tensor)
        
        test_state = lk.Tensor(private_test[it]).Reshape(dim + [1])
        cum_loss = cum_loss + lk.MeanAll(LossFunction(public_state, test_state))
        cur_tensor = join_info(test_state, hidden_state)
    cum_loss = cum_loss + lk.SumAll(cur_tensor) * 0
    return cum_loss / float(len(private_test))


start_checkpoint = time.perf_counter()
mse_loss = calculate_loss(model, lk.MSELoss)
mae_loss = calculate_loss(model, lk.MAELoss)
rmse_loss = calculate_loss(model, lk.RMSELoss)
print("No Training:")
print("MSE Loss: ", mse_loss)
print("MAE Loss: ", mae_loss)
print("RMSE Loss: ", rmse_loss)

end_checkpoint = time.perf_counter()
print("Time elapsed in inference: ", end_checkpoint - start_checkpoint, " second(s)")


model.load_state_dict(lk.ReadFile(WEIGHT_PATH))

start_checkpoint = time.perf_counter()
mse_loss = calculate_loss(model, lk.MSELoss)
mae_loss = calculate_loss(model, lk.MAELoss)
rmse_loss = calculate_loss(model, lk.RMSELoss)
print("Yes Training:")
print("MSE Loss: ", mse_loss)
print("MAE Loss: ", mae_loss)
print("RMSE Loss: ", rmse_loss)
end_checkpoint = time.perf_counter()

end_checkpoint = time.perf_counter()
print("Time elapsed in inference: ", end_checkpoint - start_checkpoint, " second(s)")