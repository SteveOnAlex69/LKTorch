#include <TensorManip/Optimizer.hpp>
#include <DebugAssist/DebugAssist.hpp>


SGD::SGD(float lr) {
	learning_rate = lr;
}
float SGD::get_learning_rate() const { return learning_rate; }
void SGD::set_learning_rate(float lr) { learning_rate = lr; }

void SGD::add_parameter(Tensor ts) { parameters.push_back(ts);}

void SGD::step() {
	for (auto i : parameters) i.gradient_descent(learning_rate);
}
void SGD::zero_gradient() {
	for (auto i : parameters) i.zero_gradient();
}