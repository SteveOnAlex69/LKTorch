#include <Helper/GeneralMath.hpp>
#include <DebugAssist/DebugAssist.hpp>
#include <Tensor/BaseTensor.hpp>

Tensor::Tensor(std::shared_ptr<RawTensor> t) { ts = t; }
Tensor::Tensor(StaticIntVector d) { ts = std::make_shared<RawTensor>(d); }
Tensor::Tensor(StaticIntVector d, std::vector<float> x) { ts = std::make_shared<RawTensor>(d, x);}
Tensor::Tensor(StaticIntVector d, StaticFloatVector x) { ts = std::make_shared<RawTensor>(d, x); }

int Tensor::get_tensor_size() {	return ts->get_tensor_size(); }
StaticIntVector Tensor::get_tensor_dimension() {return ts->get_tensor_dimension();}

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

Tensor Tensor::Transpose() { return Tensor(transpose(ts)); }
Tensor Tensor::Reshape(StaticIntVector y) { return Tensor(reshape(ts, y)); }
Tensor Tensor::Slice(StaticIntVector l, StaticIntVector r) { return Tensor(slice(ts, l, r)); }
Tensor Tensor::Merge(Tensor y) { return Tensor(merge(ts, y.ts)); }
Tensor Tensor::ValueMultiply(Tensor y) { return Tensor(value_multiply(ts, y.ts)); }