#include <TensorManip/Loss.hpp>
#include <DebugAssist/DebugAssist.hpp>
#include <Helper/GeneralMath.hpp>

Tensor MSELoss(Tensor x, Tensor y) {
	int x_size = x.get_tensor_size(), y_size = y.get_tensor_size();
	if (x_size != y_size) throw_error("Size mismatched when calculating loss");
	return Sum(Sqr(x - y));
}

Tensor MAELoss(Tensor x, Tensor y) {
	int x_size = x.get_tensor_size(), y_size = y.get_tensor_size();
	if (x_size != y_size) throw_error("Size mismatched when calculating loss");
	return Sum(Abs(x - y));
}