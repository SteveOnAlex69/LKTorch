#ifndef TENSORINIT_HPP
#define TENSORINIT_HPP

#include <Tensor/BaseTensor.hpp>

Tensor Zeros(StaticIntVector dimension);
Tensor Ones(StaticIntVector dimension);
Tensor UniformValue(StaticIntVector dimension, float x);
Tensor UniformRandom(StaticIntVector dimension, float l, float r);
Tensor NormalRandom(StaticIntVector dimension, float mean, float deviation);
Tensor CustomRandom(StaticIntVector dimension, std::function<float(float)> init_function);


#endif