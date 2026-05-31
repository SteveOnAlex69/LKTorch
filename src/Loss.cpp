#include <Loss.hpp>
#include <DebugAssist.hpp>
#include <GeneralMath.hpp>

Tensor MSELoss(Tensor x, Tensor y) {
	int x_size = x.get_tensor_size(), y_size = y.get_tensor_size();
	if (x_size != y_size) throw_error("Size mismatched when calculating loss");
	Tensor d = x - y;
	return change_dimension(d, std::vector<int>{ 1, x_size }) * change_dimension(d, std::vector<int>{ x_size, 1 });
}