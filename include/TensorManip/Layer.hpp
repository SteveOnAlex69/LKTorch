#ifndef LAYER_HPP
#define LAYER_HPP

#include <Tensor/Tensor.hpp>
#include <TensorManip/Module.hpp>
#include <vector>

class LinearLayer : public Module {
public:
	LinearLayer() { n = m = 0; W = Zeros(std::vector<int>{0}); B = Zeros(std::vector<int>{0});}
	LinearLayer(int x, int y) {
		n = x, m = y;
		W = UniformRandom(std::vector<int>{x, y}, -1, 1);
		B = UniformRandom(std::vector<int>{y}, -1, 1);
		register_parameter(W);
		register_parameter(B);
	}
	Tensor forward (Tensor x) override {return x * W + B;}
private:
	int n, m;
	Tensor W, B;
};


#endif