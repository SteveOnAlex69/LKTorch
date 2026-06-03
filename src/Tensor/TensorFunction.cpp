#include <Tensor/TensorFunction.hpp>

typedef std::function<StaticFloatVector(StaticFloatVector&)> VectorFunction;

TensorFunction::TensorFunction(VectorFunction f, VectorFunction dF) : f(f), dF(dF) {}
std::shared_ptr<RawTensor> TensorFunction::forward(std::shared_ptr<RawTensor> x) {
	std::shared_ptr<RawTensor> ans = std::make_shared<RawTensor>(x->get_tensor_dimension());
	ans->A() = std::make_shared<StaticFloatVector>(f(*x->A()));
	ans->add_parent(x); 
	ans->set_trans_type(CUSTOM_FUNCTION);
	ans->setDF(dF);
	return ans;
}
Tensor TensorFunction::forward(Tensor f) { return Tensor(forward(f.ts)); }

std::shared_ptr<RawTensor> TensorFunction::operator () (std::shared_ptr<RawTensor> x) { return forward(x); }
Tensor TensorFunction::operator () (Tensor x) { return forward(x); }

