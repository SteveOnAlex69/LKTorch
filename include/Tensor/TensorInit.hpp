#ifndef TENSORINIT_HPP
#define TENSORINIT_HPP

#include <Tensor/BaseTensor.hpp>

Tensor Zeros(std::vector<int> dimension);
Tensor Ones(std::vector<int> dimension);
Tensor UniformValue(std::vector<int> dimension, float x);
Tensor UniformRandom(std::vector<int> dimension, float l, float r);
Tensor NormalRandom(std::vector<int> dimension, float mean, float deviation);
Tensor CustomRandom(std::vector<int> dimension, std::function<float(float)> init_function);


#endif