#include <Tensor/TensorInit.hpp>
#include <Helper/GeneralMath.hpp>
#include <random>


Tensor Zeros(std::vector<int> dimension) {
	return UniformValue(dimension, 0);
}
Tensor Ones(std::vector<int> dimension) {
	return UniformValue(dimension, 1);
}
Tensor UniformValue(std::vector<int> dimension, float val) {
	int sz = 1;
	for (int i = 0; i < dimension.size(); ++i) sz *= dimension[i];
	return Tensor(dimension, std::vector<float>(sz, val));
}
Tensor UniformRandom(std::vector<int> dimension, float l, float r) {
	int sz = 1;
	for (int i = 0; i < dimension.size(); ++i) sz *= dimension[i];

	std::vector<float> suck(sz);
	for (int i = 0; i < sz; ++i) suck[i] = uniform_random_float(l, r);
	return Tensor(dimension, suck);
}
Tensor NormalRandom(std::vector<int> dimension, float mean, float deviation) {
	int sz = 1;
	for (int i = 0; i < dimension.size(); ++i) sz *= dimension[i];

	std::vector<float> suck(sz);
	for (int i = 0; i < sz; ++i) suck[i] = normal_random_float(mean, deviation);

	return Tensor(dimension, suck);
}

Tensor CustomRandom(std::vector<int> dimension, std::function<float(int)> init_function) {
	int sz = 1;
	for (int i = 0; i < dimension.size(); ++i) sz *= dimension[i];

	std::vector<float> suck(sz);
	for (int i = 0; i < sz; ++i) suck[i] = init_function(i);

	return Tensor(dimension, suck);
}