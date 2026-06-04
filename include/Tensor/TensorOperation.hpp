#ifndef TENSOROPERATION_HPP
#define TENSOROPERATION_HPP

#include <Tensor/BaseTensor.hpp>
#include <Tensor/TensorFunction.hpp>

extern TensorFunction reLU, Sigmoid, Abs, Sqr, Sqrt, Log, Tanh, Inverse, Min, Max;


Tensor Reshape(Tensor x, StaticIntVector y);
Tensor Transpose(Tensor x);
Tensor Slice(Tensor x, StaticIntVector l, StaticIntVector r);
Tensor Merge(Tensor x, Tensor y);

Tensor Sum(Tensor x);
Tensor Flatten(Tensor x);
Tensor Mean(Tensor x);
Tensor SoftMax(Tensor x);

Tensor Huber(Tensor x, float epsilon);


Tensor operator * (Tensor x, float y);
Tensor operator / (Tensor x, float y);
Tensor operator + (Tensor x, float y);
Tensor operator - (Tensor x, float y);


Tensor ScalarMultiply(Tensor x, float y);
Tensor ScalarDivide(Tensor x, float y);
Tensor ScalarAdd(Tensor x, float y);
Tensor ScalarSubtract(Tensor x, float y);

#endif