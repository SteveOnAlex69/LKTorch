 #include <Helper/GeneralMath.hpp>
#include <DebugAssist/DebugAssist.hpp>
#include <Tensor/BaseTensor.hpp>

Tensor::Tensor(std::shared_ptr<RawTensor> t) { ts = t; }
Tensor::Tensor(std::vector<int> d) { ts = std::make_shared<RawTensor>(d); }
Tensor::Tensor(std::vector<int> d, std::vector<float> x) { ts = std::make_shared<RawTensor>(d, x);}

int Tensor::get_tensor_size() {	return ts->get_tensor_size(); }
std::vector<int> Tensor::get_tensor_dimension() {
	return cast_to_vector(ts->get_tensor_dimension());
}

void Tensor::backward(bool is_root) {
	if (ts == nullptr) throw_error("You are back propogating on an empty Tensor");
	ts->backward(is_root);
	if (ts->empty()) ts = nullptr;
}

void Tensor::clear() {ts = nullptr;}
bool Tensor::empty() { return ts == nullptr; }


std::shared_ptr<StaticFloatVector>& Tensor::A() { return ts->A(); }
std::shared_ptr<StaticFloatVector>& Tensor::gA() { { return ts->gA(); } }

Tensor operator + (Tensor  x, Tensor  y) { return Tensor(x.ts + y.ts); }
Tensor operator - (Tensor  x, Tensor  y) { return Tensor(x.ts - y.ts); }
Tensor operator * (Tensor  x, Tensor  y) { return Tensor(x.ts * y.ts); }


Tensor Tensor::Permute(std::vector<int> new_d, std::vector<int> new_perm) { return Tensor(permute(ts, new_d, new_perm)); }
Tensor Tensor::PermuteDimension(std::vector<int> perm) { return Tensor(permute_dimension(ts, perm)); }
Tensor Tensor::Transpose() { return Tensor(transpose(ts)); }
Tensor Tensor::Reshape(std::vector<int> y) { return Tensor(reshape(ts, y)); }
Tensor Tensor::Slice(std::vector<int> l, std::vector<int> r) { return Tensor(slice(ts, l, r)); }
Tensor Tensor::Merge(Tensor y) { return Tensor(merge(ts, y.ts)); }
Tensor Tensor::ValueMultiply(Tensor y) { return Tensor(value_multiply(ts, y.ts)); }