#include <GeneralMath.hpp>
#include <Tensor.hpp>
#include <DebugAssist.hpp>


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

void RawTensor::add_parent(std::shared_ptr<RawTensor> p) {
	p->increase_reference_count();
	parents.push_back(p);
}
std::vector<std::shared_ptr<RawTensor>> RawTensor::get_parent() const {return parents;}

bool RawTensor::empty() { return tensor_size == 0; }


std::shared_ptr<StaticFloatVector>& RawTensor::A() { return value; }
std::shared_ptr<StaticFloatVector>& RawTensor::gA() { return gradient_value; }


float& RawTensor::accessA(int x) { return (*value)[x]; }
float& RawTensor::accessGA(int x) { return (*gradient_value)[x]; }

void RawTensor::backward() {
	if (reference_counter) return;
	for (auto i : parents) i->decrease_reference_count();
	if (parents.empty()) return;
	
	if (t_type == NOTHING) {
		
	}
	else if (t_type == ADD) {
		int x_size = parents[0]->get_tensor_size(), y_size = parents[1]->get_tensor_size();
		int total_size = get_tensor_size();
		for (int i = 0; i < total_size; ++i) {
			parents[0]->accessGA(i % x_size) += accessGA(i);
			parents[1]->accessGA(i % y_size) += accessGA(i);
		}
	}
	else if (t_type == SUBTRACT) {
		int x_size = parents[0]->get_tensor_size(), y_size = parents[1]->get_tensor_size();
		int total_size = get_tensor_size();
		for (int i = 0; i < total_size; ++i) {
			parents[0]->accessGA(i % x_size) += accessGA(i);
			parents[1]->accessGA(i % y_size) -= accessGA(i);
		}
	}
	else if (t_type == MULTIPLY) {
		int d1 = parents[0]->get_tensor_dimension()[0], d2 = parents[0]->get_tensor_dimension()[1],
			d3 = parents[1]->get_tensor_dimension()[1];

		for (int x = 0; x < d1; ++x)
			for (int y = 0; y < d2; ++y)
				for (int z = 0; z < d3; ++z)
					parents[0]->accessGA(x * d2 + y) += accessGA(x * d3 + z) * parents[1]->accessA(y * d3 + z);
		for (int x = 0; x < d1; ++x)
			for (int y = 0; y < d2; ++y)
				for (int z = 0; z < d3; ++z)
					parents[1]->accessGA(y * d3 + z) += accessGA(x * d3 + z) * parents[0]->accessA(x * d2 + y);
	}
	else if (t_type == RELU) {
		int x_size = get_tensor_size();
		for (int i = 0; i < x_size; ++i)
			parents[0]->accessGA(i) += accessGA(i) * DreLU(parents[0] -> accessA(i));
	}
	else if (t_type == SIGMOID) {
		int x_size = get_tensor_size();
		for (int i = 0; i < x_size; ++i)
			parents[0]->accessGA(i) += accessGA(i) * Dsigmoid(parents[0]->accessA(i));
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
	for (int i = 0; i < max_size; ++i)
		ans->accessA(i) = x->accessA(i % x_size) + y->accessA(i % y_size);

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
	for (int i = 0; i < max_size; ++i)
		ans->accessA(i) = x->accessA(i % x_size) - y->accessA(i % y_size);

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
	for (int i = 0; i < dih1[0]; ++i)
		for (int k = 0; k < dih1[1]; ++k)
			for (int j = 0; j < dih2[1]; ++j)
				ans->accessA(i * dih3[1] + j) += x->accessA(i * dih1[1] + k) * y->accessA(k * dih2[1] + j);
	ans->add_parent(x);ans->add_parent(y);
	ans->set_trans_type(MULTIPLY);
	return ans;
}
std::shared_ptr<RawTensor> change_dimension(std::shared_ptr<RawTensor> x, StaticIntVector y) {
	int new_size = 1;
	for (int i = 0; i < y.size(); ++i) new_size *= y[i];
	if (new_size != x->get_tensor_size()) throw_error("Inappropriate tensor size when changing dimension");

	std::shared_ptr<RawTensor> ans = std::make_shared<RawTensor>(y);
	ans->A() = x->A();
	ans->gA() = x->gA();
	ans->add_parent(x); 
	return ans;
}


std::shared_ptr<RawTensor> reLU(std::shared_ptr<RawTensor> x) {
	int x_size = x->get_tensor_size();
	std::shared_ptr<RawTensor> ans = std::make_shared<RawTensor>(x->get_tensor_dimension());
	for (int i = 0; i < x_size; ++i)
		ans->accessA(i) = reLU(x->accessA(i));

	ans->add_parent(x);
	ans->set_trans_type(RELU);
	return ans;
}
std::shared_ptr<RawTensor> sigmoid(std::shared_ptr<RawTensor> x) {
	int x_size = x->get_tensor_size();
	std::shared_ptr<RawTensor> ans = std::make_shared<RawTensor>(x->get_tensor_dimension());
	for (int i = 0; i < x_size; ++i)
		ans->accessA(i) = sigmoid(x->accessA(i));

	ans->add_parent(x);
	ans->set_trans_type(SIGMOID);
	return ans;
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

void Tensor::backward() {
	if (ts == nullptr) throw_error("You are back propogating on an empty Tensor");
	ts->backward();
	if (ts->empty()) ts = nullptr;
}

void Tensor::clear() {ts = nullptr;}
bool Tensor::empty() { return ts == nullptr; }


std::shared_ptr<StaticFloatVector>& Tensor::A() { return ts->A(); }
std::shared_ptr<StaticFloatVector>& Tensor::gA() { { return ts->gA(); } }


Tensor operator + (Tensor  x, Tensor  y) { return Tensor(x.ts + y.ts); }
Tensor operator - (Tensor  x, Tensor  y) { return Tensor(x.ts - y.ts); }
Tensor operator * (Tensor  x, Tensor  y) { return Tensor(x.ts * y.ts); }
Tensor change_dimension(Tensor x, StaticIntVector y) { return Tensor(change_dimension(x.ts, y)); }


Tensor reLU(Tensor x) { return Tensor(reLU(x.ts)); }
Tensor sigmoid(Tensor x) { return Tensor(sigmoid(x.ts)); }


void Tensor::gradient_descent(float lr) { ts->gradient_descent(lr); }
void Tensor::zero_gradient() { ts->zero_gradient(); }
void Tensor::multiply_gradient(float x) { ts->multiply_gradient(x); }


float& Tensor::accessA(int x) { return ts->accessA(x); }
float& Tensor::accessGA(int x) { return ts->accessGA(x); }

std::ostream& operator << (std::ostream& os, Tensor x) {
	StaticIntVector d = x.get_tensor_dimension();
	for (int i = d.size() - 2; i >= 0; --i) d[i] *= d[i + 1];
	for (int i = 0; i < x.get_tensor_size(); ++i) {
		for (int j = 0; j < d.size(); ++j) if (i % d[j] == 0) os << "(";
		os << x.accessA(i);
		int count = 0;
		for (int j = 0; j < d.size(); ++j) if ((i+1) % d[j] == 0) os << ")";
		if (i + 1 < x.get_tensor_size())
			os << ", ";
	}
	return os;
}
