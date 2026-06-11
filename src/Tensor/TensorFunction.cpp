#include <Tensor/TensorFunction.hpp>

typedef std::function<std::vector<float>(std::vector<float>)> VectorFunction;

TensorFunction::TensorFunction() {}
TensorFunction::TensorFunction(VectorFunction f, VectorFunction dF) : f(f), dF(dF) {}
std::shared_ptr<RawTensor> TensorFunction::forward(std::shared_ptr<RawTensor> x) {
	std::shared_ptr<RawTensor> ans = std::make_shared<RawTensor>(x->get_tensor_dimension());
	std::vector<float> tmp = f(cast_to_vector(*x->A()));
	for (int i = 0; i < (int)tmp.size(); ++i)
		ans->A()->data()[i] = tmp[i];
	ans->add_parent(x); 
	ans->set_trans_type(CUSTOM_FUNCTION);
	ans->setDF(dF);
	return ans;
}
Tensor TensorFunction::forward(Tensor f) { return Tensor(forward(f.ts)); }

std::shared_ptr<RawTensor> TensorFunction::operator () (std::shared_ptr<RawTensor> x) { return forward(x); }
Tensor TensorFunction::operator () (Tensor x) { return forward(x); }