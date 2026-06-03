#include <Tensor/TensorInit.hpp>
#include <Helper/GeneralMath.hpp>
#include <random>


Tensor Zeros(StaticIntVector dimension) {
	return UniformValue(dimension, 0);
}
Tensor Ones(StaticIntVector dimension) {
	return UniformValue(dimension, 1);
}
Tensor UniformValue(StaticIntVector dimension, float val) {
	return UniformRandom(dimension, val, val);
}
Tensor UniformRandom(StaticIntVector dimension, float l, float r) {
	int sz = 1;
	for (int i = 0; i < dimension.size(); ++i) sz *= dimension[i];

	StaticFloatVector suck(sz);
	for (int i = 0; i < sz; ++i) suck[i] = uniform_random_float(l, r);
	return Tensor(dimension, suck);
}
Tensor NormalRandom(StaticIntVector dimension, float mean, float deviation) {
	int sz = 1;
	for (int i = 0; i < dimension.size(); ++i) sz *= dimension[i];

	StaticFloatVector suck(sz);
	for (int i = 0; i < sz; ++i) suck[i] = normal_random_float(mean, deviation);

	return Tensor(dimension, suck);
}