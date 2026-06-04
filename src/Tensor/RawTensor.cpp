#include <Helper/GeneralMath.hpp>
#include <DebugAssist/DebugAssist.hpp>
#include <Tensor/RawTensor.hpp>


RawTensor::RawTensor(StaticIntVector d) {
	dimension = d;
	tensor_size = 1;
	for (int i = 0; i < dimension.size(); ++i) tensor_size *= dimension[i];
	A() = std::make_shared<StaticFloatVector>(tensor_size);
	gA() = std::make_shared<StaticFloatVector>(tensor_size);
	reference_counter = 0;
	t_type = NOTHING;
}

RawTensor::RawTensor(StaticIntVector d, std::shared_ptr<StaticFloatVector> preA, std::shared_ptr<StaticFloatVector> pregA) {
	dimension = d;
	tensor_size = 1;
	for (int i = 0; i < dimension.size(); ++i) tensor_size *= dimension[i];
	A() = preA;
	gA() = pregA;
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
int RawTensor::get_tensor_size() {return tensor_size;}
StaticIntVector RawTensor::get_tensor_dimension() {return dimension;}


RawTensor::~RawTensor() { 
	clear(); 
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

void RawTensor::set_start_dimension(StaticIntVector d) { start_dimension = d; }

void RawTensor::setDF(std::function<StaticFloatVector(StaticFloatVector&)> f) { dF = f; }

void RawTensor::add_parent(std::shared_ptr<RawTensor> p) {
	p->increase_reference_count();
	parents.push_back(p);
}
std::vector<std::shared_ptr<RawTensor>> RawTensor::get_parent() const { return parents; }

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
	if (is_root)
		for (int j = 0; j < cur_gA.size(); ++j)
			cur_gA[j] += 1;

	for (auto i : parents) i->decrease_reference_count();
	if (parents.empty()) return;

	if (t_type == NOTHING) {
	}
	else if (t_type == ADD) {
		const int x_size = parents[0]->get_tensor_size(), y_size = parents[1]->get_tensor_size();
		int total_size = get_tensor_size();

		float* __restrict parent0_gA = parents[0]->gA()->data();
		float* __restrict parent1_gA = parents[1]->gA()->data();
		const float* __restrict cur_gA = gA()->data();
		for (int i = 0; i < total_size; ++i) {
			parent0_gA[i % x_size] += cur_gA[i];
			parent1_gA[i % y_size] += cur_gA[i];
		}
	}
	else if (t_type == SUBTRACT) {
		const int x_size = parents[0]->get_tensor_size(), y_size = parents[1]->get_tensor_size();
		int total_size = get_tensor_size();
		float* __restrict parent0_gA = parents[0]->gA()->data();
		float* __restrict parent1_gA = parents[1]->gA()->data();
		const float* __restrict cur_gA = gA()->data();
		for (int i = 0; i < total_size; ++i) {
			parent0_gA[i % x_size] += cur_gA[i];
			parent1_gA[i % y_size] -= cur_gA[i];
		}
	}
	else if (t_type == MULTIPLY) {
		const int d2 = parents[1]->get_tensor_dimension()[0];
		const int d1 = parents[0]->get_tensor_size() / d2, d3 = parents[1]->get_tensor_size() / d2;
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
		float* __restrict parent0_gA = parents[0]->gA()->data();
		const float* __restrict cur_gA = gA()->data();
		for (int i = 0; i < x_size; ++i)
			parent0_gA[i] += cur_gA[i] * tmp[i];
	}
	else if (t_type == TRANSPOSE) {
		StaticIntVector d = get_tensor_dimension();
		int d1 = d[0], d2 = d[1];

		const float* __restrict cur_gA = gA()->data();
		float* __restrict parent0_gA = parents[0]->gA()->data();
		for (int i = 0; i < d1; ++i)
			for (int j = 0; j < d2; ++j)
				parent0_gA[j * d1 + i] += cur_gA[i * d2 + j];
	}
	else if (t_type == SLICE) {
		const float* __restrict cur_gA = gA()->data();
		float* __restrict parent0_gA = parents[0]->gA()->data();
		
		StaticIntVector l = start_dimension;
		StaticIntVector d = parents[0]->get_tensor_dimension();
		StaticIntVector new_d = get_tensor_dimension();
		int sz = get_tensor_size();
		for (int i = 0; i < sz; ++i) {
			int prod1 = 1;
			int idx = 0, tmp = i;
			for (int j = new_d.size() - 1; j >= 0; --j) {
				idx += (tmp % new_d[j] + l[j]) * prod1;
				tmp /= new_d[j];
				prod1 *= d[j];
			}
			parent0_gA[idx] += cur_gA[i];
		}
	}
	else if (t_type == MERGE) {
		const float* __restrict cur_gA = gA()->data();
		float* __restrict parent0_gA = parents[0]->gA()->data();
		float* __restrict parent1_gA = parents[1]->gA()->data();

		const int sz1 = parents[0]->get_tensor_size(), sz2 = parents[1]->get_tensor_size();
		for (int i = 0; i < sz1; ++i) parent0_gA[i] += cur_gA[i];
		for (int i = 0; i < sz2; ++i) parent1_gA[i] += cur_gA[i + sz1];
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

	const int x_size = x->get_tensor_size(), y_size = y->get_tensor_size();
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

	const int x_size = x->get_tensor_size(), y_size = y->get_tensor_size();
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
	if (dih1.size() == 0 || dih2.size() == 0) throw_error("Cannot perform tensor multiplication with rank-0 tensor");
	if (dih1[dih1.size() - 1] != dih2[0]) throw_error("Mismatch size when tensor multiplication");

	StaticIntVector dih3(dih1.size() + dih2.size() - 2);	
	for (int i = 0; i + 1 < dih1.size(); ++i) dih3[i] = dih1[i];
	for (int i = 0; i + 1 < dih2.size(); ++i) dih3[dih1.size() + i - 1] = dih2[i+1];

	std::shared_ptr<RawTensor> ans = std::make_shared<RawTensor>(dih3);
	const int d1 = x->get_tensor_size() / dih2[0], d2 = dih2[0], d3 = y->get_tensor_size() / dih2[0];
	const float* __restrict parent0_A = x->A()->data();
	const float* __restrict parent1_A = y->A()->data();
	float* __restrict cur_A = ans->A()->data();
	for (int i = 0; i < d1; ++i)
		for (int k = 0; k < d2; ++k)
			for (int j = 0; j < d3; ++j)
				cur_A[i * d3 + j] += parent0_A[i * d2 + k] * parent1_A[k * d3 + j];
	ans->add_parent(x);ans->add_parent(y);
	ans->set_trans_type(MULTIPLY);
	return ans;
}
std::shared_ptr<RawTensor> reshape(std::shared_ptr<RawTensor> cur, StaticIntVector y) {
	StaticIntVector x = cur->get_tensor_dimension();
	int new_size = 1;
	for (int i = 0; i < y.size(); ++i) new_size *= y[i];
	if (new_size != cur->get_tensor_size()) throw_error("Inappropriate tensor size when changing dimension");


	// checking if the two size are equivalent, using two pointers
	int i = 0, j = 0;
	while (i < x.size() && j < y.size()) {
		if (x[i] == 1) {i++;continue;}
		if (y[j] == 1) {j++;continue;}
		int prod1 = x[i++], prod2 = y[j++];
		while (prod1 < prod2 && prod2 % prod1 == 0) prod1 *= x[i++];
		while (prod1 > prod2 && prod1 % prod2 == 0) prod2 *= y[j++];
		i++; j++;
		if (prod1 != prod2) 
			throw_error("The structure when reshaping are not similar (you can only either slice the new dimension into parts, or join the old dimensions into a big one)");
	}

	std::shared_ptr<RawTensor> ans = std::make_shared<RawTensor>(y, cur->A(), cur->gA());
	ans->add_parent(cur);
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
	return ans;
}
std::shared_ptr<RawTensor> slice(std::shared_ptr<RawTensor> x, StaticIntVector l, StaticIntVector r) {
	StaticIntVector d = x->get_tensor_dimension();
	if (d.size() != l.size() || d.size() != r.size())
		throw_error("Mismatched dimension when slicing!");
	for (int i = 0; i < d.size(); ++i) {
		if (l[i] > r[i])
			throw_error("L[i] > R[i] when slicing!");
		if (l[i] < 0) throw_error("L[i] < 0");
		if (r[i] >= d[i]) throw_error("R[i] >= D[i]");
	}

	StaticIntVector new_d = d;
	for (int i = 0; i < d.size(); ++i) new_d[i] = r[i] - l[i] + 1;
	std::shared_ptr<RawTensor> ans = std::make_shared<RawTensor>(new_d);
	float* curA = x->A()->data();
	float* nextA = ans->A()->data();

	int sz = ans->get_tensor_size();
	for (int i = 0; i < sz; ++i) {
		int prod1 = 1;
		int idx = 0, tmp = i;
		for (int j = new_d.size() - 1; j >= 0; --j) {
			idx += (tmp % new_d[j] + l[j]) * prod1;
			tmp /= new_d[j];
			prod1 *= d[j];
		}
		nextA[i] = curA[idx];
	}

	ans->add_parent(x);
	ans->set_trans_type(SLICE);
	ans->set_start_dimension(l);

	return ans;
}
std::shared_ptr<RawTensor> merge(std::shared_ptr<RawTensor> x, std::shared_ptr<RawTensor> y) {
	StaticIntVector d1 = x->get_tensor_dimension(), d2 = y->get_tensor_dimension();
	if (d1.size() != d2.size()) throw_error("Mismatched dimension when merging tensor");
	for(int i = 1; i < d1.size(); ++i) if (d1[i] != d2[i])
		throw_error("Mismatched dimension when merging tensor");

	StaticIntVector d = d1;
	d[0] += d2[0];
	std::shared_ptr<RawTensor> ans = std::make_shared<RawTensor>(d);
	float* xA = x->A()->data();	
	float* yA = y->A()->data();
	float* nextA = ans->A()->data();
	const int sz1 = x->get_tensor_size(), sz2 = y->get_tensor_size();

	for (int i = 0; i < sz1; ++i) nextA[i] = xA[i];
	for (int i = 0; i < sz2; ++i) nextA[i + sz1] = yA[i];

	ans->add_parent(x);
	ans->add_parent(y);
	ans->set_trans_type(MERGE);
	return ans;
}

void RawTensor::gradient_descent(float lr) {*value -= *gradient_value * lr;}
void RawTensor::zero_gradient() { multiply_gradient(0); }
void RawTensor::multiply_gradient(float x) { *gradient_value *= x; }