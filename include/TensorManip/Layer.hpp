#ifndef LAYER_HPP
#define LAYER_HPP

#include <Tensor/Tensor.hpp>
#include <vector>

class LinearLayer {
public:
	LinearLayer(int x, int y);
	void randomize_weight_normal(float mean = 0, float deviation = 1);
	void randomize_weight_uniform(float l = -1, float r = 1);

	Tensor forward(Tensor x);
	Tensor operator () (Tensor x);

	std::vector<Tensor> get_parameters();
private:
	int n, m;
	Tensor W, B;
};


#endif