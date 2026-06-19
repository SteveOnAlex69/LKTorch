#include <Tensor/SingletonTensor.hpp>

namespace TensorSingleton {
	bool backprop = true;
}

void activate_backprop() { TensorSingleton::backprop = true; }
void deactivate_backprop() { TensorSingleton::backprop = false; }
void toggle_backprop() { TensorSingleton::backprop = !TensorSingleton::backprop; }
void set_backprop_status(bool f) { TensorSingleton::backprop = f; }
bool backprop_activated() {return TensorSingleton::backprop;}


namespace DebugSingleton {
	bool debug = false;
}

void activate_debug() { DebugSingleton::debug = true; }
void deactivate_debug() { DebugSingleton::debug = false; }
bool debug_activated() { return DebugSingleton::debug; }