#ifndef TENSORMATHOP_HPP
#define TENSORMATHOP_HPP


#include <Tensor/BaseTensor.hpp>
#include <Tensor/TensorFunction.hpp>

extern TensorFunction reLU, Sigmoid, Abs, Sqr, Sqrt, Cube, Cbrt, Log, Tanh, Inverse, Min, Max, Exp;
extern TensorFunction Sin, Cos, Tan, Asin, Acos, Atan;

Tensor Huber(Tensor x, float epsilon);


#endif