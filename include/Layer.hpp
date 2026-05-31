#ifndef LAYER_HPP
#define LAYER_HPP

#include <Tensor.hpp>
#include <vector>

class LinearLayer {
public:
	LinearLayer(int x, int y);
	void randomize_weight();

	Tensor forward(Tensor x);
	Tensor operator () (Tensor x);

	std::vector<Tensor> get_parameters();
private:
	int n, m;
	Tensor W, B;
};


#endif