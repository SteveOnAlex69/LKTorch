# YO
# Welcome to the Libary Introduction
# This example will display how the library training and inference process would look like.
# Here, we will try to approximate the function f(t) = sin(t)
# But there is a catch: We are not given the entire function at once. 
# We are only given a data point, f(t), and we have to predict f(t + 0.01)
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

public_test = []
for i in range(0, 100):
    current_test = []
    for j in range(0, 10):
        current_test.append(math.sin(math.pi * (i + j) / 50))
    public_test.append(current_test)

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

end_checkpoint = time.perf_counter()
print("Time elapsed in data fetching: ", end_checkpoint - start_checkpoint, " second(s)")

start_checkpoint = time.perf_counter()

optimizer = lk.Adam(0.001, 0.9, 0.999)
optimizer.add_parameter(model.get_parameters())

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

def calculate_loss(model):
    input_data = lk.Tensor(public_test[0])
    dim = input_data.dimension()
    input_data = input_data.Reshape(dim + [1])
    input_data = join_info(input_data, lk.Zeros(dim + [3]))

    cur_tensor = input_data
    cum_loss = lk.Zeros([])
    for it in range(1, len(public_test)):
        cur_tensor = model(cur_tensor)
        public_state = extract_public(cur_tensor)
        hidden_state = extract_hidden(cur_tensor)
        
        test_state = lk.Tensor(public_test[it]).Reshape(dim + [1])
        cum_loss = cum_loss + lk.MeanAll(lk.MSELoss(public_state, test_state))
        cur_tensor = join_info(test_state, hidden_state)
    cum_loss = cum_loss + lk.SumAll(cur_tensor) * 0
    return cum_loss

print("before:", calculate_loss(model))

for epoch in range(1, 501):
    for segment in range(0, 10):
        l = 1
        if segment > 0:
            l = segment * 10
        r = segment * 10 + 9

        optimizer.zero_gradient()

        input_data = lk.Tensor(public_test[0])
        dim = input_data.dimension()
        input_data = input_data.Reshape(dim + [1])
        input_data = join_info(input_data, lk.Zeros(dim + [3]))

        cur_tensor = input_data
        cum_loss = lk.Zeros([])

        for it in range(1, l):
            cur_tensor = model(cur_tensor)
            public_state = extract_public(cur_tensor)
            hidden_state = extract_hidden(cur_tensor)
            
            test_state = lk.Tensor(public_test[it]).Reshape(dim + [1])
            cur_tensor = join_info(test_state, hidden_state)
        for it in range(l, r+1):
            cur_tensor = model(cur_tensor)
            public_state = extract_public(cur_tensor)
            hidden_state = extract_hidden(cur_tensor)
            
            test_state = lk.Tensor(public_test[it]).Reshape(dim + [1])
            cum_loss = cum_loss + lk.MeanAll(lk.MSELoss(public_state, test_state))
            cur_tensor = join_info(test_state, hidden_state)
        cum_loss = cum_loss + lk.SumAll(cur_tensor) * 0
        cum_loss.backward()
        optimizer.step()

    if (epoch % 10 == 9):
        print("Epoch: ", epoch)
        print("--- Loss: ", calculate_loss(model))


end_checkpoint = time.perf_counter()
print("Time elapsed in training: ", end_checkpoint - start_checkpoint, " second(s)")


lk.WriteFile(model.get_state_dict(), WEIGHT_PATH)
print("File saved successfully!")