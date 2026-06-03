#ifndef TENSOROPERATION_HPP
#define TENSOROPERATION_HPP

#include <Tensor/BaseTensor.hpp>
#include <Tensor/TensorFunction.hpp>

extern TensorFunction reLU, Sigmoid, Abs, Sqr, Sqrt, Log, Min, Max;

Tensor Sum(Tensor x);
Tensor Flatten(Tensor x);
Tensor operator * (Tensor x, float y);
Tensor operator / (Tensor x, float y);

#endif