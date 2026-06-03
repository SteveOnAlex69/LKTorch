#include <Helper/GeneralMath.hpp>
#include <Tensor/BaseTensor.hpp>
#include <DebugAssist/DebugAssist.hpp>


RawTensor::RawTensor(StaticIntVector d) {
	dimension = d;
	tensor_size = 1;
	for (int i = 0; i < dimension.size(); ++i) tensor_size *= dimension[i];
	A() = std::make_shared<StaticFloatVector>(tensor_size); 
	gA() = std::make_shared<StaticFloatVector>(tensor_size);
	reference_counter = 0;
	t_type = NOTHING;
}

RawTensor::RawTensor(StaticIntVector d, StaticFloatVector x) {
	dimension = d;
	tensor_size = 1;
	for (int i = 0; i < dimension.size(); ++i) tensor_size *= dimension[i];

	A() = std::make_shared<StaticFloatVector>(x); gA() = std::make_shared<StaticFloatVector>(tensor_size);
	reference_counter = 0;
	t_type = NOTHING;
}

RawTensor::RawTensor(StaticIntVector d, std::vector<float> x) : RawTensor(d, StaticFloatVector(x)) {}


int RawTensor::get_tensor_size() {
	return tensor_size;
}
StaticIntVector RawTensor::get_tensor_dimension() {
	return dimension;
}

void RawTensor::clear() {
	tensor_size = 0;
	A() = nullptr;
	gA() = nullptr;

	dimension.clear();
	reference_counter = 0;
	parents.clear();
	t_type = NOTHING;
}

int RawTensor::get_reference_count() {
	return reference_counter;
}
void RawTensor::increase_reference_count() {
	reference_counter++;
}
void RawTensor::decrease_reference_count() {
	reference_counter--;
}

void RawTensor::set_trans_type(TransformType trans) { t_type = trans; }
TransformType RawTensor::get_trans_type() const { return t_type; }
void RawTensor::setDF(std::function<StaticFloatVector(StaticFloatVector&)> f) { dF = f; }

void RawTensor::add_parent(std::shared_ptr<RawTensor> p) {
	p->increase_reference_count();
	parents.push_back(p);
}
std::vector<std::shared_ptr<RawTensor>> RawTensor::get_parent() const {return parents;}

bool RawTensor::empty() { return tensor_size == 0; }


std::shared_ptr<StaticFloatVector>& RawTensor::A() { return value; }
std::shared_ptr<StaticFloatVector>& RawTensor::gA() { return gradient_value; }


float& RawTensor::accessA(StaticIntVector x) {
	if (x.size() != dimension.size()) throw_error("Dimension mismatched when accessing A of tensor");
	int idx = 0, prod = 1;
	for (int j = dimension.size() - 1; j >= 0; --j) {
		idx += x[j] * prod;
		prod *= dimension[j];
	}
	return (*A())[idx];
}
float& RawTensor::accessGA(StaticIntVector x) {
	if (x.size() != dimension.size()) throw_error("Dimension mismatched when accessing gA of tensor");
	int idx = 0, prod = 1;
	for (int j = dimension.size() - 1; j >= 0; --j) {
		idx += x[j] * prod;
		prod *= dimension[j];
	}
	return (*gA())[idx];
}

void RawTensor::backward(bool is_root) {
	if (reference_counter) return;

	StaticFloatVector& cur_gA = *gA();
	if (is_root) {
		for (int j = 0; j < cur_gA.size(); ++j)
			cur_gA[j] += 1;
	}

	for (auto i : parents) i->decrease_reference_count();
	if (parents.empty()) return;
	
	if (t_type == NOTHING) {
		StaticFloatVector& parent0_gA = *parents[0]->gA();
		StaticFloatVector& cur_gA = *gA();
		parent0_gA += cur_gA;
	}
	else if (t_type == ADD) {
		const int x_size = parents[0]->get_tensor_size(), y_size = parents[1]->get_tensor_size();
		int total_size = get_tensor_size();

		StaticFloatVector& parent0_gA = *parents[0]->gA();
		StaticFloatVector& parent1_gA = *parents[1]->gA();
		StaticFloatVector& cur_gA = *gA();
		for (int i = 0; i < total_size; ++i) {
			parent0_gA[i % x_size] += cur_gA[i];
			parent1_gA[i % y_size] += cur_gA[i];
		}
	}
	else if (t_type == SUBTRACT) {
		const int x_size = parents[0]->get_tensor_size(), y_size = parents[1]->get_tensor_size();
		int total_size = get_tensor_size();
		StaticFloatVector& parent0_gA = *parents[0]->gA();
		StaticFloatVector& parent1_gA = *parents[1]->gA();
		StaticFloatVector& cur_gA = *gA();
		for (int i = 0; i < total_size; ++i) {
			parent0_gA[i % x_size] += cur_gA[i];
			parent1_gA[i % y_size] -= cur_gA[i];
		}
	}
	else if (t_type == MULTIPLY) {
		const int d1 = parents[0]->get_tensor_dimension()[0], d2 = parents[0]->get_tensor_dimension()[1],
			d3 = parents[1]->get_tensor_dimension()[1];

		const float* __restrict cur_gA = gA()->data();
		const float* __restrict parent0_A = parents[0]->A()->data();
		const float* __restrict parent1_A = parents[1]->A()->data();
		float* __restrict parent0_gA = parents[0]->gA()->data();
		float* __restrict parent1_gA = parents[1]->gA()->data();
		for (int x = 0; x < d1; ++x)
			for (int y = 0; y < d2; ++y)
				for (int z = 0; z < d3; ++z) {
					parent0_gA[x * d2 + y] += cur_gA[x * d3 + z] * parent1_A[y * d3 + z];
					parent1_gA[y * d3 + z] += cur_gA[x * d3 + z] * parent0_A[x * d2 + y];
				}
	}
	else if (t_type == CUSTOM_FUNCTION) {
		int x_size = get_tensor_size();
		StaticFloatVector& tmp = dF(*parents[0]->A());
		StaticFloatVector& cur_gA = *gA();
		StaticFloatVector& parent0_gA = *parents[0]->gA();
		for (int i = 0; i < x_size; ++i)
			parent0_gA[i] += cur_gA[i] * tmp[i];
	}
	else if (t_type == TRANSPOSE) {
		StaticIntVector d = get_tensor_dimension();
		int d1 = d[0], d2 = d[1];

		const float* cur_gA = gA()->data();
		float* parent0_gA = parents[0]->gA()->data();
		for (int i = 0; i < d1; ++i)
			for (int j = 0; j < d2; ++j)
				parent0_gA[j * d1 + i] += cur_gA[i * d2 + j];
	}
	else throw_error("unknown transition type");

	for (auto i : parents)
		i->backward();

	clear();
}


std::shared_ptr<RawTensor> operator + (std::shared_ptr<RawTensor> x, std::shared_ptr<RawTensor> y) {
	StaticIntVector& dx = x->get_tensor_dimension(), dy = y->get_tensor_dimension();
	int min_size = std::min(dx.size(), dy.size());
	for (int i = 0; i < min_size; ++i)
		if (dx[dx.size() - min_size + i] != dy[dy.size() - min_size + i])
			throw_error("dimension mismatch");
	
	int x_size = x->get_tensor_size(), y_size = y->get_tensor_size();
	int max_size = std::max(x_size, y_size);
	StaticIntVector max_d = (dx.size() > dy.size()) ? x->get_tensor_dimension() : y->get_tensor_dimension();
	std::shared_ptr<RawTensor> ans = std::make_shared<RawTensor>(max_d);


	StaticFloatVector& parent0_A = *x->A();
	StaticFloatVector& parent1_A = *y->A();
	StaticFloatVector& cur_A = *ans->A();
	for (int i = 0; i < max_size; ++i)
		cur_A[i] = parent0_A[i % x_size] + parent1_A[i % y_size];

	ans->add_parent(x); ans->add_parent(y);
	ans->set_trans_type(ADD);
	return ans;
}
std::shared_ptr<RawTensor> operator - (std::shared_ptr<RawTensor> x, std::shared_ptr<RawTensor> y) {
	StaticIntVector& dx = x->get_tensor_dimension(), dy = y->get_tensor_dimension();
	int min_size = std::min(dx.size(), dy.size());
	for (int i = 0; i < min_size; ++i)
		if (dx[dx.size() - min_size + i] != dy[dy.size() - min_size + i])
			throw_error("dimension mismatch");

	int x_size = x->get_tensor_size(), y_size = y->get_tensor_size();
	int max_size = std::max(x_size, y_size);
	StaticIntVector max_d = (dx.size() > dy.size()) ? x->get_tensor_dimension() : y->get_tensor_dimension();
	std::shared_ptr<RawTensor> ans = std::make_shared<RawTensor>(max_d);
	StaticFloatVector& parent0_A = *x->A();
	StaticFloatVector& parent1_A = *y->A();
	StaticFloatVector& cur_A = *ans->A();
	for (int i = 0; i < max_size; ++i)
		cur_A[i] = parent0_A[i % x_size] - parent1_A[i % y_size];

	ans->add_parent(x); ans->add_parent(y);
	ans->set_trans_type(SUBTRACT);
	return ans;
}
std::shared_ptr<RawTensor> operator * (std::shared_ptr<RawTensor> x, std::shared_ptr<RawTensor> y) {
	StaticIntVector dih1 = x->get_tensor_dimension(), dih2 = y->get_tensor_dimension();
	if (dih1.size() != 2 || dih2.size() != 2) throw_error("I can only multiply two rank 2 tensor bro");
	if (dih1[1] != dih2[0]) throw_error("Mismatch size when matrix multiplication");

	StaticIntVector dih3 = std::vector<int>{dih1[0], dih2[1]};
	std::shared_ptr<RawTensor> ans = std::make_shared<RawTensor>(dih3);

	const float* __restrict parent0_A = x->A()->data();
	const float* __restrict parent1_A = y->A()->data();
	float* __restrict cur_A = ans->A()->data();

	const int d1 = dih1[0], d2 = dih1[1], d3 = dih2[1];

	for (int i = 0; i < d1; ++i)
		for (int k = 0; k < d2; ++k)
			for (int j = 0; j < d3; ++j)
				cur_A[i * d3 + j] += parent0_A[i * d2 + k] * parent1_A[k * d3 + j];
	ans->add_parent(x);ans->add_parent(y);
	ans->set_trans_type(MULTIPLY);
	return ans;
}
std::shared_ptr<RawTensor> change_dimension(std::shared_ptr<RawTensor> x, StaticIntVector y) {
	int new_size = 1;
	for (int i = 0; i < y.size(); ++i) new_size *= y[i];
	if (new_size != x->get_tensor_size()) throw_error("Inappropriate tensor size when changing dimension");

	std::shared_ptr<RawTensor> ans = std::make_shared<RawTensor>(y);

	StaticFloatVector tmp = *x->A();
	ans->A() = std::make_shared<StaticFloatVector>(tmp);
	ans->add_parent(x); 
	return ans;
}
std::shared_ptr<RawTensor> transpose(std::shared_ptr<RawTensor> x) {
	StaticIntVector d = x->get_tensor_dimension();
	if (d.size() != 2) throw_error("Transpose called on a tensor with non rank 2");

	const int d1 = d[0], d2 = d[1];

	std::shared_ptr<RawTensor> ans = std::make_shared<RawTensor>(std::vector<int>{d2, d1});
	float* curA = x->A()->data();
	float* nextA = ans->A()->data();

	for (int i = 0; i < d1; ++i)
		for (int j = 0; j < d2; ++j)
			nextA[j * d1 + i] = curA[i * d2 + j];
	
	ans->add_parent(x);
	ans->set_trans_type(TRANSPOSE);
}



void RawTensor::gradient_descent(float lr) {
	*value -= *gradient_value * lr;
}
void RawTensor::zero_gradient() { multiply_gradient(0); }
void RawTensor::multiply_gradient(float x) {*gradient_value *= x;}

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
Tensor ChangeDimension(Tensor x, StaticIntVector y) { return Tensor(change_dimension(x.ts, y)); }
Tensor Transpose(Tensor x) { return Tensor(transpose(x.ts)); }


void Tensor::gradient_descent(float lr) { ts->gradient_descent(lr); }
void Tensor::zero_gradient() { ts->zero_gradient(); }
void Tensor::multiply_gradient(float x) { ts->multiply_gradient(x); }
Tensor Tensor::Transpose() {return Tensor(transpose(ts) );}
Tensor Tensor::ChangeDimension(StaticIntVector y) {return Tensor(change_dimension(ts, y));}



std::ostream& operator << (std::ostream& os, Tensor x) {
	StaticIntVector d = x.get_tensor_dimension();
	StaticFloatVector& A = *(x.A());
	for (int i = d.size() - 2; i >= 0; --i) d[i] *= d[i + 1];
	for (int i = 0; i < x.get_tensor_size(); ++i) {
		for (int j = 0; j < d.size(); ++j) if (i % d[j] == 0) os << "(";
		os << A[i];
		int count = 0;
		for (int j = 0; j < d.size(); ++j) if ((i+1) % d[j] == 0) os << ")";
		if (i + 1 < x.get_tensor_size())
			os << ", ";
	}
	return os;
}
