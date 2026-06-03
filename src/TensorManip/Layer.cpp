#include <TensorManip/Layer.hpp>
#include <Helper/GeneralMath.hpp>
#include <Tensor/Tensor.hpp>

LinearLayer::LinearLayer(int x, int y) {
	n = x, m = y;
	W = Zeros(std::vector<int>{x, y});
	B = Zeros(std::vector<int>{y});
}
void LinearLayer::randomize_weight_normal(float mean, float deviation) {
	W = NormalRandom(std::vector<int>{n, m}, mean, deviation);
	B = NormalRandom(std::vector<int>{m}, mean, deviation);
}


void LinearLayer::randomize_weight_uniform(float l, float r) {
	W = UniformRandom(std::vector<int>{n, m}, l, r);
	B = UniformRandom(std::vector<int>{m}, l, r);
}

Tensor LinearLayer::forward(Tensor x) {
	return x * W + B;
}
Tensor LinearLayer::operator () (Tensor x) {
	return forward(x);
}

std::vector<Tensor> LinearLayer::get_parameters() {
	return std::vector<Tensor>{W, B};
}