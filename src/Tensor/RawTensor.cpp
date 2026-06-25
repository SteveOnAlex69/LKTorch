#include <Helper/GeneralMath.hpp>
#include <DebugAssist/DebugAssist.hpp>
#include <Tensor/RawTensor.hpp>


RawTensor::RawTensor(StaticIntVector d, bool nograd) {
	dimension = d;
	tensor_size = 1;
	for (int i = 0; i < dimension.size(); ++i) tensor_size *= dimension[i];
	A() = std::make_shared<StaticFloatVector>(tensor_size);
	if (nograd) gA() = nullptr;
	else gA() = std::make_shared<StaticFloatVector>(tensor_size);
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


RawTensor::RawTensor(StaticIntVector d, StaticFloatVector x, bool nograd) {
	dimension = d;
	tensor_size = 1;
	for (int i = 0; i < dimension.size(); ++i) tensor_size *= dimension[i];

	A() = std::make_shared<StaticFloatVector>(x);
	if (nograd) gA() = nullptr;
	else gA() = std::make_shared<StaticFloatVector>(tensor_size);
	reference_counter = 0;
	t_type = NOTHING;
}

RawTensor::RawTensor(StaticIntVector d, std::vector<float> x, bool nograd) : RawTensor(d, StaticFloatVector(x), nograd) {}
int RawTensor::get_tensor_size() { return tensor_size; }
StaticIntVector RawTensor::get_tensor_dimension() { return dimension; }
int RawTensor::get_row() {
	if (dimension.size() == 0) return 1;
	return dimension[dimension.size() - 1];
}

RawTensor::~RawTensor() { 
	if (reference_counter) throw_error("Tensor destroyed while still having reference counter");
	if (empty()) clear();
	else backward(false);
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

void RawTensor::set_df(std::function<std::vector<float>(std::vector<float>, int)>  f) { dF = f; }

void RawTensor::add_parent(std::shared_ptr<RawTensor> p) {
	p->increase_reference_count();
	parents.push_back(p);
}
std::vector<std::shared_ptr<RawTensor>> RawTensor::get_parent() const { return parents; }
bool RawTensor::empty() { return tensor_size == 0; }

std::shared_ptr<StaticFloatVector>& RawTensor::A() { return value; }
std::shared_ptr<StaticFloatVector>& RawTensor::gA() { return gradient_value; }
void RawTensor::set_permutation(std::vector<std::pair<int, int>> p, std::pair<int, int> s) {
	perm = p;
	stride = s;
}


float& RawTensor::access_A(StaticIntVector x) {
	if (x.size() != dimension.size()) throw_error("Dimension mismatched when accessing A of tensor");
	int idx = 0, prod = 1;
	for (int j = dimension.size() - 1; j >= 0; --j) {
		idx += x[j] * prod;
		prod *= dimension[j];
	}
	return (*A())[idx];
}
float& RawTensor::access_GA(StaticIntVector x) {
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

	if (gA() == nullptr) 
		gA() = std::make_shared<StaticFloatVector>(get_tensor_size());

	StaticFloatVector& cur_gA = *gA();
	if (is_root)
		for (int j = 0; j < cur_gA.size(); ++j)
			cur_gA[j] += 1;

	if (debug_activated()) {
		std::cout << this << " " << t_type << "\n";
		std::cout << (*A()) << "\n";
		std::cout << cur_gA << "\n";
		std::cout << parents << "\n";
	}

	for (auto i : parents) i->decrease_reference_count();
	if (parents.empty()) return;

	for (auto i : parents) {
		if (i -> gA() == nullptr)
			i->gA() = std::make_shared<StaticFloatVector>(i->get_tensor_size());
	}
	if (t_type == NOTHING) {
		int total_size = get_tensor_size();

		float* __restrict parent0_gA = parents[0]->gA()->data();
		const float* __restrict cur_gA = gA()->data();
		for (int i = 0; i < total_size; ++i) {
			parent0_gA[i] += cur_gA[i];
		}
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
	else if (t_type == VALUE_MULTIPLY) {
		const int x_size = parents[0]->get_tensor_size(), y_size = parents[1]->get_tensor_size();
		int total_size = get_tensor_size();
		float* __restrict parent0_gA = parents[0]->gA()->data();
		float* __restrict parent1_gA = parents[1]->gA()->data();
		const float* __restrict cur_gA = gA()->data();
		const float* __restrict parent0_A = parents[0]->A()->data();
		const float* __restrict parent1_A = parents[1]->A()->data();
		for (int i = 0; i < total_size; ++i) {
			parent0_gA[i % x_size] += cur_gA[i] * parent1_A[i % y_size];
			parent1_gA[i % y_size] += cur_gA[i] * parent0_A[i % x_size];
		}
	}
	else if (t_type == CUSTOM_FUNCTION) {
		int x_size = get_tensor_size();
		int row = 1;
		StaticIntVector dim = get_tensor_dimension();
		if (dim.size() > 0) row = dim[dim.size() - 1];
		
		std::vector<float> tmp = dF(cast_to_vector(*parents[0]->A()), get_row());
		float* __restrict parent0_gA = parents[0]->gA()->data();
		const float* __restrict cur_gA = gA()->data();
		for (int i = 0; i < x_size; ++i)
			parent0_gA[i] += cur_gA[i] * tmp[i];
	}
	else if (t_type == CUSTOM_PERMUTE) {
		int x_size = get_tensor_size();
		const float* __restrict cur_gA = gA()->data();
		float* __restrict parent0_gA = parents[0]->gA()->data();

		int p1 = 0, p2 = 0;
		while (p2 < x_size) {
			for (std::pair<int, int> i : perm)
				parent0_gA[i.first + p1] += cur_gA[i.second + p2];

			p1 += stride.first;
			p2 += stride.second;
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

	while (parents.size()) 
		parents.pop_back();
	clear();
}


std::shared_ptr<RawTensor> operator + (std::shared_ptr<RawTensor> x, std::shared_ptr<RawTensor> y) {
	StaticIntVector dx = x->get_tensor_dimension(), dy = y->get_tensor_dimension();
	int min_size = std::min(dx.size(), dy.size());
	for (int i = 0; i < min_size; ++i)
		if (dx[dx.size() - min_size + i] != dy[dy.size() - min_size + i])
			throw_error("Tensor addition: dimension mismatch");

	const int x_size = x->get_tensor_size(), y_size = y->get_tensor_size();
	int max_size = std::max(x_size, y_size);
	StaticIntVector max_d = (dx.size() > dy.size()) ? x->get_tensor_dimension() : y->get_tensor_dimension();
	std::shared_ptr<RawTensor> ans = std::make_shared<RawTensor>(max_d, true);


	const float* __restrict parent0_A = x->A()->data();
	const float* __restrict parent1_A = y->A()->data();
	float* __restrict cur_A = ans->A()->data();
	for (int i = 0; i < max_size; ++i)
		cur_A[i] = parent0_A[i % x_size] + parent1_A[i % y_size];

	ans->add_parent(x); ans->add_parent(y);
	ans->set_trans_type(ADD);
	return ans;
}
std::shared_ptr<RawTensor> operator - (std::shared_ptr<RawTensor> x, std::shared_ptr<RawTensor> y) {
	StaticIntVector dx = x->get_tensor_dimension(), dy = y->get_tensor_dimension();
	int min_size = std::min(dx.size(), dy.size());
	for (int i = 0; i < min_size; ++i)
		if (dx[dx.size() - min_size + i] != dy[dy.size() - min_size + i])
			throw_error("Tensor subtraction: dimension mismatch");

	const int x_size = x->get_tensor_size(), y_size = y->get_tensor_size();
	int max_size = std::max(x_size, y_size);
	StaticIntVector max_d = (dx.size() > dy.size()) ? x->get_tensor_dimension() : y->get_tensor_dimension();
	std::shared_ptr<RawTensor> ans = std::make_shared<RawTensor>(max_d, true);


	const float* __restrict parent0_A = x->A()->data();
	const float* __restrict parent1_A = y->A()->data();
	float* __restrict cur_A = ans->A()->data();
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

	std::shared_ptr<RawTensor> ans = std::make_shared<RawTensor>(dih3, true);
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

std::shared_ptr<RawTensor> value_multiply(std::shared_ptr<RawTensor> x, std::shared_ptr<RawTensor> y) {
	StaticIntVector dx = x->get_tensor_dimension(), dy = y->get_tensor_dimension();
	int min_size = std::min(dx.size(), dy.size());
	for (int i = 0; i < min_size; ++i)
		if (dx[dx.size() - min_size + i] != dy[dy.size() - min_size + i])
			throw_error("Tensor value multiplication: dimension mismatch");

	const int x_size = x->get_tensor_size(), y_size = y->get_tensor_size();
	int max_size = std::max(x_size, y_size);
	StaticIntVector max_d = (dx.size() > dy.size()) ? x->get_tensor_dimension() : y->get_tensor_dimension();
	std::shared_ptr<RawTensor> ans = std::make_shared<RawTensor>(max_d, true);


	const float* __restrict parent0_A = x->A()->data();
	const float* __restrict parent1_A = y->A()->data();
	float* __restrict cur_A = ans->A()->data();
	for (int i = 0; i < max_size; ++i)
		cur_A[i] = parent0_A[i % x_size] * parent1_A[i % y_size];

	ans->add_parent(x); ans->add_parent(y);
	ans->set_trans_type(VALUE_MULTIPLY);
	return ans;
}

std::shared_ptr<RawTensor> reshape(std::shared_ptr<RawTensor> cur, StaticIntVector y) {
	int new_size = 1;
	for (int i = 0; i < y.size(); ++i) new_size *= y[i];

	StaticIntVector x = cur->get_tensor_dimension();
	int tot1 = 1;
	int idx = -1;
	if (new_size == tot1) idx = x.size();
	for (int i = x.size() - 1; i >= 0; --i) {
		tot1 *= x[i];
		if (tot1 == new_size) idx = i;
	}

	if (idx == -1) throw_error("Invalid size when reshaping");


	// checking if the two size are equivalent, using two pointers
	int i = idx, j = 0;
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

	StaticIntVector ans_dih(idx + y.size());
	for (int i = 0; i < idx; ++i) ans_dih[i] = x[i];
	for (int i = 0; i < y.size(); ++i) ans_dih[i + idx] = y[i];

	std::shared_ptr<RawTensor> ans = std::make_shared<RawTensor>(ans_dih, cur->A(), nullptr);
	ans->add_parent(cur);
	return ans;
}



std::shared_ptr<RawTensor> map_value(std::shared_ptr<RawTensor> x, std::vector<std::pair<int, int>> perm, 
	std::pair<int, int> stride, StaticIntVector new_dimension) {
	int size1 = x->get_tensor_size(), size2 = 1;
	for (int i = 0; i < new_dimension.size(); ++i)
		size2 *= new_dimension[i];

	if (stride.first == 0 || stride.second == 0 || size1 % stride.first || size2 % stride.second 
		|| size1 / stride.first * stride.second != size2) {
		throw_error("Inappropriate stride size in RawTensor::map_value");
	}

	for (std::pair<int, int> i : perm) {
		if (i.first < 0 || i.second < 0 || i.first >= stride.first || i.second >= stride.second)
			throw_error("Invalid perm in RawTensor::map_value");
	}

	std::shared_ptr<RawTensor> ans = std::make_shared<RawTensor>(new_dimension, true);
	float* curA = x->A()->data();
	float* nextA = ans->A()->data();



	int p1 = 0, p2 = 0;
	while (p1 < size1) {
		for (std::pair<int, int> i : perm)
			nextA[i.second + p2] += curA[i.first + p1];
		p1 += stride.first;
		p2 += stride.second;
	}

	ans->add_parent(x);
	ans->set_permutation(perm, stride);
	ans->set_trans_type(CUSTOM_PERMUTE);
	return ans;
}

std::shared_ptr<RawTensor> permute(std::shared_ptr<RawTensor> x, StaticIntVector d, StaticIntVector perm) {
	StaticIntVector x_dih = x->get_tensor_dimension();

	int tot1 = 1;
	int idx = -1;
	for (int i = x_dih.size() - 1; i >= 0; --i) {
		tot1 *= x_dih[i];
		if (tot1 == perm.size()) idx = i;
	}

	int tot2 = 1;
	for (int i = 0; i < d.size(); ++i) tot2 *= d[i];
	if (idx == -1 || tot2 != perm.size()) throw_error("Permute failed, the dimension mismatched");

	StaticIntVector cur_d(d.size() + idx);
	for (int i = 0; i < idx; ++i) cur_d[i] = x_dih[i];
	for (int i = 0; i < d.size(); ++i) cur_d[i + idx] = d[i];

	std::shared_ptr<RawTensor> ans = std::make_shared<RawTensor>(cur_d, true);
	float* curA = x->A()->data();
	float* nextA = ans->A()->data();

	for (int i = 0; i < tot1; ++i)
		nextA[i - i % perm.size() + perm[i % perm.size()]] = curA[i];

	ans->add_parent(x);

	std::pair<int, int> stride(perm.size(), perm.size());
	std::vector<std::pair<int, int>> poo;
	for (int i = 0; i < perm.size(); ++i)
		poo.push_back(std::make_pair(i, perm[i]));
	
	ans->set_permutation(poo, stride);
	ans->set_trans_type(CUSTOM_PERMUTE);
	return ans;
}
std::shared_ptr<RawTensor> permute_dimension(std::shared_ptr<RawTensor> x, StaticIntVector d_arg) {
	StaticIntVector x_dih = x->get_tensor_dimension();
	if (d_arg.size() > x_dih.size()) throw_error("Dimension count mismatched!");

	StaticIntVector d(x_dih.size());
	for (int i = 0; i < x_dih.size() - d_arg.size(); ++i)
		d[i] = i;
	for (int i = 0; i < d_arg.size(); ++i)
		d[i + (x_dih.size() - d_arg.size())] = d_arg[i] + x_dih.size() - d_arg.size();

	StaticIntVector dih(d.size());
	for (int i = 0; i < d.size(); ++i) dih[i] = x_dih[d[i]];

	int sz = x->get_tensor_size();
	StaticIntVector perm(sz);
	for (int i = 0; i < sz; ++i) {
		int cur = i;
		StaticIntVector fuck(x_dih.size());
		for (int i = x_dih.size() - 1; i >= 0; --i) {
			fuck[i] = cur % x_dih[i];
			cur /= x_dih[i];
		}

		int target = 0;
		for (int i = 0; i < d.size(); ++i) {
			target = target * dih[i] + fuck[d[i]];
		}
		perm[i] = target;
	}

	return permute(x, dih, perm);
}
std::shared_ptr<RawTensor> transpose(std::shared_ptr<RawTensor> x) {
	StaticIntVector d = x->get_tensor_dimension();
	if (d.size() < 2) throw_error("Transpose called on a tensor with rank less than 2");
	return permute_dimension(x, StaticIntVector({ 1, 0 }));
}
std::shared_ptr<RawTensor> slice(std::shared_ptr<RawTensor> x, StaticIntVector _l, StaticIntVector _r) {
	StaticIntVector d = x->get_tensor_dimension();
	if (_l.size() != _r.size() || d.size() < _l.size()) throw_error("Mismatched dimension when slicing!");
	StaticIntVector l(d.size()), r(d.size());
	for (int i = 0; i < d.size() - _l.size(); ++i) {
		l[i] = 0;
		r[i] = d[i] - 1;
	}

	int stride1 = 1, stride2 = 1;
	for (int i = 0; i < _l.size(); ++i) {
		stride1 *= d[i + (d.size() - _l.size())];
		stride2 *= (_r[i] - _l[i] + 1);
		l[i + (d.size() - _l.size())] = _l[i];
		r[i + (d.size() - _l.size())] = _r[i];
	}


	for (int i = 0; i < d.size(); ++i) {
		if (l[i] > r[i])
			throw_error("L[i] > R[i] when slicing!");
		if (l[i] < 0) throw_error("L[i] < 0");
		if (r[i] >= d[i]) throw_error("R[i] >= D[i]");
	}

	StaticIntVector new_d = d;
	for (int i = 0; i < d.size(); ++i) new_d[i] = r[i] - l[i] + 1;

	std::vector<std::pair<int,int>> pepe;
	std::pair<int, int> stride(stride1, stride2);

	StaticIntVector perm(std::vector<int>{0});
	for (int i = 0; i < d.size(); ++i) {
		StaticIntVector poo(perm.size() * new_d[i]);
		int idx = 0;
		for (int j = 0; j < perm.size(); ++j) {
			for (int k = l[i]; k <= r[i]; ++k)
				poo[idx++] = perm[j] * d[i] + k;
		}
		perm = poo;
	}

	for (int i = 0; i < perm.size(); ++i) pepe.push_back({ perm[i], i });

	return map_value(x, pepe, stride, new_d);
}
std::shared_ptr<RawTensor> merge(std::shared_ptr<RawTensor> x, std::shared_ptr<RawTensor> y) {
	StaticIntVector d1 = x->get_tensor_dimension(), d2 = y->get_tensor_dimension();
	if (d1.size() != d2.size()) throw_error("Mismatched dimension when merging tensor");
	for(int i = 1; i < d1.size(); ++i) if (d1[i] != d2[i])
		throw_error("Mismatched dimension when merging tensor");

	StaticIntVector d = d1;
	d[0] += d2[0];
	std::shared_ptr<RawTensor> ans = std::make_shared<RawTensor>(d, true);
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

void RawTensor::set_name(std::string s) { name = s; }
std::string RawTensor::get_name() { return name; }