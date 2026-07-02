![alt text](image-1.png)

LKTorch is a Python package that provides two high-level features:

- Tensor computation (like NumPy) with no GPU acceleration!!
- Deep neural networks built on a tape-based autograd system

Essentially, this is a hobby project of a college student trying to clone Pytorch from scratch!

This library is compatible with numpy (converting from numpy array to lk.Tensor, and vice versa)

## More about LKTorch

LKTorch is a library that consist of the following elements:

|Components|Description|
|---|---|
|lk.Tensor|The base tensor class|
|lk.TensorFunction|Differentiable function that will be apply on each row of the tensor|
|lk.TensorMathOp|Standard , differentiable math functions for machine learning, such as reLU, log, ...|
|lk.TensorOperations|Other functions for tensor manipulation, like Slice, Unfold, ...|
|lk.Module|Compacting a complex sequence of tensor manipulation into a class|
|lk.Layer|Class that manipulate tensors, such as LinearLayer, Conv2D, Dropout, ...., inherited from lk.Module|
|lk.Optimizer|A collection of popular optimizer, such as GradientDescent, Adam, ...|


### A tensor library

You may ask: what is a tensor, is it edible, and why is everyone talking about it in Machine Learning? 

When we started programming, we learn how to manipulate variable around, from simple operations like addition, multiplication, to the more complex ones like GCD.

However, there is not much fun thing that can be solved with singular variables. So there comes the concept of an array. An array is a list of variables, therefore it is exponentially more interesting.

But array is still not enough to solve problems in programming. So we invented array of arrays, array of array of arrays of arrays. Informally, these would be called 2D and 3D array, respectively.

A tensor is just a genelizations of them. It is a multi-dimentional array. A rank-1 tensor is an array, rank-2 is a matrix, rank-0 is a variable (scalar), etc.

LKTorch offers way to manipulate tensors around. Obviously, it is much weaker than Pytorch and Numpy in this regard, because it was made by a bored college student. But I do offer a way to convert to numpy array.

### Dynamic Neural Network

<details>
<summary>Prerequisite Knowledge:</summary>

<details>
<summary>Prerequisite knowledge 1:</summary> 

*A neural network is a structure that takes in a tensor, and spit out a tensor.*
</details>


<details>
<summary>Prerequisite knowledge 2:</summary> 

*Most of the time, one or more tensor will be involved in the structure of the neural network to perform linear transform like tensor addition, multiplication. However, in its untrained state, neural network usually spit out bullcrap, which means we have to  use loss function to punish it.*
</details>


<details>
<summary>Prerequisite knowledge 3:</summary> 

*You can make the neural network learn your dataset, by calculating the gradient of the entire network with respect to the loss function. Then, you gradient descent, which means you modify the tensor value with the goal of making the loss smaller.*
</details>

</details>


Other libraries, like TensorFlow, force you to build the neural network before hand, and only then, could you start to use that to manipulate data and train it. This static behavior sometimes restrict you.

With some pointer blackmagic, LKTorch allows you to build the neural network on the fly, while still allowing you to calculate the gradient of every tensor involved in the network. In essence, you can just treat `Tensor` as another variable, where you can perform addition, multiplication, log, etc., and at the end of the day, call `.backward()`, and it will calculate the gradient of the entire ordeal for you. 

This means you can literally put if statement inside your neural network training. Here is an example.

```python
x = model1(x)
# Model 1 has a probability to spit out crap, so if it indeed spit out crap, just put the same thing into model 2
if (x.numpy().max() < 10):
	x = model2(x) 

# Now we can calculate the gradient of that whole thing
x.backward()
```

Is LKTorch the only library that does this? Yes it is! This is a totally original library! There is no other library with a similar name, let alone a similar functionality.

### My Sanity First

I will ~~parody~~ copy [Pytorch README.md](https://github.com/pytorch/pytorch?tab=readme-ov-file#more-about-pytorch) on this one

LKTorch is a Python binding into a monolithic C++ framework. I tried to build it to be deeply integrated into Python, and that turned out half correct. You can kinda use it naturally like you would use NumPy / SciPy / scikit-learn etc, and you can even convert the object into numpy array! You can write your new neural network layers in Python itself. Our goal is to  reinvent the wheel.

### Imperative Experiences

This is designed to be easy to use (I hope so).

### Framework Used

What?

<details>
<summary>Serious</summary>

In all seriousness, this was more of a learning project than an actual production code, so I didn't use any helping library aside from **pybind11** to bind my code into python.
</details>

## How to install?

### On Window:

Simply type into Command Prompt (cmd)
```
pip install lktorch
```

and we are done!

### On OS that is not Window:

What?

## How to install from repo


### On Window:

Simply type into Command Prompt (cmd)
```
pip install -e .
```

and we are done!

### On OS that is not Window:

What?

## My thought process

If you are interested on my thought process, and how this project was conceived, check this out: [How did I build LKTorch?](docs/HOW.md)