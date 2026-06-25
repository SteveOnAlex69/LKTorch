#ifndef TENSORMATHOP_HPP
#define TENSORMATHOP_HPP


#include <Tensor/BaseTensor.hpp>
#include <Tensor/TensorFunction.hpp>

extern TensorFunction reLU, Sigmoid, Abs;
extern TensorFunction Min, Max;
extern TensorFunction Sqr, Sqrt, Cube, Cbrt;
extern TensorFunction Log, Inverse, Exp;
extern TensorFunction Sinh, Cosh, Tanh;
extern TensorFunction Sin, Cos, Tan, Asin, Acos, Atan;

Tensor Huber(Tensor x, float epsilon);
Tensor Pow(Tensor x, float p);



#endif