#include <Layer.hpp>
#include <GeneralMath.hpp>

LinearLayer::LinearLayer(int x, int y) {
	n = x, m = y;
	W = Tensor(std::vector<int>{x, y});
	B = Tensor(std::vector<int>{y});
}
void LinearLayer::randomize_weight() {
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < m; ++j)
			W.accessA(i * m + j) = random_float(-1, 1);
	for (int j = 0; j < m; ++j)
		B.accessA(j) = random_float(-1, 1);
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