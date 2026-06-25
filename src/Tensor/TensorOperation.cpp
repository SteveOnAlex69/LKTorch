#include <Tensor/TensorOperation.hpp>
#include <Tensor/TensorInit.hpp>
#include <DebugAssist/DebugAssist.hpp>
#include <Tensor/DebugTensor.hpp>
#include <Helper/GeneralMath.hpp>


namespace reLU_function { // No explosion
	float forward(float x) { return (x >= 0) ? x : 0; }
	float backward(float x) { return (x >= 0) ? 1 : 0; }
}

namespace sigmoid_function { // No explosion
	float forward(float x) { 
		if (x >= 0)
			return 1.0f / (1 + std::exp(-x)); 
		return 1 - 1.0f / (1 + std::exp(x));
	}
	float backward(float x) { 
		float s = forward(x);
		return s * (1 - s);
	}
}

namespace abs_function { // No explosion
	float forward(float x) { return (x >= 0) ? x : -x; }
	float backward(float x) { return (x >= 0) ? 1 : -1; }
}

namespace sqr_function { // No explosion
	float forward(float x) { return x * x; }
	float backward(float x) { return x * 2; }
}

namespace sqrt_function { // Yes explosion
	const float EPS = 1e-8;
	const float SEPS = std::sqrt(EPS);
	float forward(float x) { 
		if (x >= EPS) return std::sqrt(x);
		return SEPS + (x - EPS) * (0.5 / SEPS);

	}
	float backward(float x) { 
		if (x >= EPS) return 0.5 / std::sqrt(x);
		return 0.5 / SEPS;
	}
}

namespace log_function { // Yes explosion
	const float EPS = 1e-8;
	const float SEPS = std::log(EPS);
	float forward(float x) {
		if (x >= EPS) return std::log(x);
		return SEPS + (x - EPS) * (1 / EPS);
	}
	float backward(float x) {
		if (x >= EPS) return 1 / x;
		return 1 / EPS;
	}
}

namespace tanh_function {
	float forward(float x) { return std::tanh(x); }
	float backward(float x) { return 1 - sqr_function::forward(forward(x)); }
}

namespace inverse_function {
	const float EPS = 1e-5;
	const float SEPS = 1 / EPS;
	float forward(float x) { 
		if (x >= 0) {
			if (x >= EPS) return 1 / x;
			return SEPS - (x - EPS) * (1 / (EPS * EPS));
		}
		else return -forward(-x);
	}
	float backward(float x) { 
		if (x >= 0) {
			if (x >= EPS) return -1 / (x * x);
			return -1 / (EPS * EPS);
		}
		else return -backward(-x);
	}
}


namespace exp_function {
	float forward(float x) {
		return std::exp(x);
	}
	float backward(float x) {
		return std::exp(x);
	}
}


TensorFunction reLU(
	[](std::vector<float> x, int row) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = reLU_function::forward(x[i]);
		return ans;
	},

	[](std::vector<float> x, int row) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = reLU_function::backward(x[i]);
		return ans;
	}
);
TensorFunction Sigmoid(
	[](std::vector<float> x, int row) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = sigmoid_function::forward(x[i]);
		return ans;
	},

	[](std::vector<float> x, int row) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = sigmoid_function::backward(x[i]);
		return ans;
	}
);
TensorFunction Abs(
	[](std::vector<float> x, int row) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = abs_function::forward(x[i]);
		return ans;
	},

	[](std::vector<float> x, int row) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = abs_function::backward(x[i]);
		return ans;
	}
);
TensorFunction Sqr(
	[](std::vector<float> x, int row) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = sqr_function::forward(x[i]);
		return ans;
	},

	[](std::vector<float> x, int row) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = sqr_function::backward(x[i]);
		return ans;
	}
);
TensorFunction Sqrt(
	[](std::vector<float> x, int row) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = sqrt_function::forward(x[i]);
		return ans;
	},

	[](std::vector<float> x, int row) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = sqrt_function::backward(x[i]);
		return ans;
	}
);
TensorFunction Log(
	[](std::vector<float> x, int row) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = log_function::forward(x[i]);
		return ans;
	},

	[](std::vector<float> x, int row) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = log_function::backward(x[i]);
		return ans;
	}
);

TensorFunction Tanh(
	[](std::vector<float> x, int row) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = tanh_function::forward(x[i]);
		return ans;
	},

	[](std::vector<float> x, int row) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = tanh_function::backward(x[i]);
		return ans;
	}
);


TensorFunction Inverse(
	[](std::vector<float> x, int row) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = inverse_function::forward(x[i]);
		return ans;
	},

	[](std::vector<float> x, int row) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = inverse_function::backward(x[i]);
		return ans;
	}
);
TensorFunction Exp(
	[](std::vector<float> x, int row) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = exp_function::forward(x[i]);
		return ans;
	},

	[](std::vector<float> x, int row) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = exp_function::backward(x[i]);
		return ans;
	}
);

TensorFunction Max(
	[](std::vector<float> x, int row) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int p = 0; p < x.size(); p += row) {
			float ma = -1e18;
			int idx = 0;
			int l = p, r = min(x.size(), p + row) - 1;
			for (int i = l; i <= r; ++i)
				if (ma < x[i]) {
					ma = x[i];
					idx = i;
				}
			ans[idx] = ma;
		}
		return ans;
	},

	[](std::vector<float> x, int row) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int p = 0; p < x.size(); p += row) {
			float ma = -1e18;
			int idx = 0;
			int l = p, r = min(x.size(), p + row) - 1;
			for (int i = l; i <= r; ++i)
				if (ma < x[i]) {
					ma = x[i];
					idx = i;
				}
			ans[idx] = 1;
		}
		return ans;
	}
);


TensorFunction Min(
	[](std::vector<float> x, int row) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int p = 0; p < x.size(); p += row) {
			float mi = 1e18;
			int idx = 0;
			int l = p, r = min(x.size(), p + row) - 1;
			for (int i = l; i <= r; ++i)
				if (mi > x[i]) {
					mi = x[i];
					idx = i;
				}
			ans[idx] = mi;
		}
		return ans;
	},

	[](std::vector<float> x, int row) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int p = 0; p < x.size(); p += row) {
			float mi = 1e18;
			int idx = 0;
			int l = p, r = min(x.size(), p + row) - 1;
			for (int i = l; i <= r; ++i)
				if (mi > x[i]) {
					mi = x[i];
					idx = i;
				}
			ans[idx] = 1;
		}
		return ans;
	}
);

Tensor MapValue(Tensor x, std::vector<std::pair<int, int>> mp, std::pair<int, int> stride, std::vector<int> new_dimension) {
	return x.MapValue(mp, stride, new_dimension);
}
Tensor Permute(Tensor x, std::vector<int> new_d, std::vector<int> new_p) { return x.Permute(new_d, new_p); }
Tensor PermuteDimension(Tensor x, std::vector<int> perm) { return x.PermuteDimension(perm); }
Tensor Transpose(Tensor x) { return x.Transpose();; }
Tensor Reshape(Tensor x, std::vector<int> y) { return x.Reshape(y); }
Tensor Slice(Tensor x, std::vector<int> l, std::vector<int> r) { return x.Slice(l, r); }
Tensor Merge(Tensor x, Tensor y) { return x.Merge(y); }
Tensor ValueMultiply(Tensor x, Tensor y) { return x.ValueMultiply(y); }


Tensor MaxPool(Tensor x) {return Sum(Max(x));}
Tensor MinPool(Tensor x) {return Sum(Min(x));}

Tensor MaxIndex(Tensor x) {
	if (x.dimension().size() == 0) throw_error("MaxIndex doesn't accept tensor of rank 0");
	std::vector<float> v = cast_to_vector(*x.A());
	int row = x.dimension().back();
	std::vector<float> ans(v.size() / row);
	for (int p = 0; p < (int)ans.size(); ++p) {
		int l = p * row, r = (p + 1) * row;
		int idx = 0;
		float cur = v[l];
		for (int i = l + 1; i < r; ++i) {
			if (cur < v[i]) {
				cur = v[i];
				idx = i - l;
			}
		}
		ans[p] = idx;
	}
	std::vector<int> result_dim = x.dimension(); result_dim.pop_back();
	return Tensor(result_dim, ans);
}


Tensor MinIndex(Tensor x) {
	if (x.dimension().size() == 0) throw_error("MaxIndex doesn't accept tensor of rank 0");
	std::vector<float> v = cast_to_vector(*x.A());
	int row = x.dimension().back();
	std::vector<float> ans(v.size() / row);
	for (int p = 0; p < (int)ans.size(); ++p) {
		int l = p * row, r = (p + 1) * row;
		int idx = 0;
		float cur = v[l];
		for (int i = l + 1; i < r; ++i) {
			if (cur > v[i]) {
				cur = v[i];
				idx = i - l;
			}
		}
		ans[p] = idx;
	}
	std::vector<int> result_dim = x.dimension(); result_dim.pop_back();
	return Tensor(result_dim, ans);
}

Tensor Sum(Tensor x) {
	StaticIntVector dx = x.dimension();
	if (dx.size() == 0) throw_error("In Sum Function: Cannot perform on rank-0 tensor");
	Tensor tmp = x * UniformValue(std::vector<int>{dx[dx.size() - 1]}, 1);
	return tmp;
}
Tensor Mean(Tensor x) {
	StaticIntVector dx = x.dimension();
	if (dx.size() == 0) throw_error("In Mean Function: Cannot perform on rank-0 tensor");
	Tensor tmp = x * UniformValue(std::vector<int>{dx[dx.size() - 1]}, (float)1 / dx[dx.size() - 1]);
	return tmp;
}

Tensor Flatten(Tensor x, int dim) {
	std::vector<int> dih = x.dimension();
	if (dim == 0) dim = dih.size();
	if (dim > dih.size()) dim = dih.size();
	
	int mul = 1;
	for (int i = 0; i < dim; ++i) {
		mul *= dih.back();
		dih.pop_back();
	}
	dih.push_back(mul);

	return x.Reshape(dih);
}


Tensor SumAll(Tensor x) { return Sum(Flatten(x)); }
Tensor MeanAll(Tensor x) { return Mean(Flatten(x)); }


Tensor operator * (Tensor x, float y) {
	return TensorFunction(
		[=](std::vector<float> x, int row) -> std::vector<float> {
			std::vector<float> ans(x.size());
			for (int i = 0; i < x.size(); ++i)
				ans[i] = x[i] * y;
			return ans;
		},

		[=](std::vector<float> x, int row) -> std::vector<float> {
			std::vector<float> ans(x.size());
			for (int i = 0; i < x.size(); ++i)
				ans[i] = y;
			return ans;
		}
	)(x);
}
Tensor operator / (Tensor x, float y) {
	return TensorFunction(
		[=](std::vector<float> x, int row) -> std::vector<float> {
			std::vector<float> ans(x.size());
			for (int i = 0; i < x.size(); ++i)
				ans[i] = x[i] / y;
			return ans;
			},

		[=](std::vector<float> x, int row) -> std::vector<float> {
			std::vector<float> ans(x.size());
			for (int i = 0; i < x.size(); ++i)
				ans[i] = 1.0/y;
			return ans;
			}
	)(x);
}


Tensor operator + (Tensor x, float y) {
	return x + Tensor(std::vector<int>{}, std::vector<float>{y});
}
Tensor operator - (Tensor x, float y) {
	return x - Tensor(std::vector<int>{}, std::vector<float>{y});
}

Tensor ScalarMultiply(Tensor x, float y) { return x * y; }
Tensor ScalarDivide(Tensor x, float y) { return x / y; }
Tensor ScalarAdd(Tensor x, float y) { return x + y; }
Tensor ScalarSubtract(Tensor x, float y) { return x - y; }

Tensor SoftMax(Tensor x) {
	std::vector<int> d = x.dimension();
	if (d.size() == 0) throw_error("In SoftMax Function: Cannot perform on rank-0 tensor");

	Tensor y = Inverse(Sum(x));
	std::vector<int> fi = { (int)d.size() - 1 };
	for (int i = 0; i + 1 < d.size(); ++i) fi.push_back(i);

	std::vector<int> se = {};
	for (int i = 0; i + 1 < d.size(); ++i) se.push_back(i + 1);
	se.push_back(0);
	
	x = PermuteDimension(x, fi);
	x = x.ValueMultiply(y);
	x = PermuteDimension(x, se);
	return x;
}


Tensor Huber(Tensor x, float epsilon) {
	return TensorFunction(
		[=](std::vector<float> x, int row) -> std::vector<float> {
			std::vector<float> ans(x.size());
			for (int i = 0; i < x.size(); ++i) {
				if (abs(x[i]) >= epsilon) ans[i] = abs(x[i]) * epsilon - 0.5 * epsilon * epsilon;
				else ans[i] = 0.5 * sqr_function::forward(x[i]);
			}
			return ans;
		},

		[=](std::vector<float> x, int row) -> std::vector<float> {
			std::vector<float> ans(x.size());
			for (int i = 0; i < x.size(); ++i) {
				if (abs(x[i]) >= epsilon)
					ans[i] = abs_function::backward(x[i]) * epsilon;
				else
					ans[i] = x[i];
			}
			return ans;
		}
	)(x);
}


Tensor Unfold(Tensor x, std::vector<int> slider_d, std::vector<int> stride, std::vector<int> modulo) {
	while (stride.size() < slider_d.size()) stride.push_back(1);
	while (modulo.size() < slider_d.size()) modulo.push_back(0);

	std::vector<int> dimension = x.dimension();
	if (dimension.size() < slider_d.size()) throw_error("Slider dimension invalid in Tensor::Unfold");

	int s1 = 1, s2 = 1;
	std::vector<int> new_dimension;
	for (int i = 0; i < dimension.size() - slider_d.size(); ++i)
		new_dimension.push_back(dimension[i]);


	for (int i = 0; i < (int)slider_d.size(); ++i) {
		int cur_d = dimension[dimension.size() - slider_d.size() + i];
		s1 *= cur_d;
		int cur_span = (cur_d - modulo[i] - slider_d[i] + stride[i]) / stride[i];
		s2 *= cur_span;
		new_dimension.push_back(cur_span);
	}

	int s3 = 1;
	for (int i : slider_d) {
		s3 *= i;
		new_dimension.push_back(i);
	}

	std::vector<std::pair<int, int>> perm;
	perm.push_back({0, 0});

	for (int i = 0; i < (int)slider_d.size(); ++i) {
		int p = dimension.size() - slider_d.size() + i;
		int cur_span = (dimension[p] - modulo[i] - slider_d[i] + stride[i]) / stride[i];
		std::vector<std::pair<int, int>> lmao;
		for (auto j : perm) {
			for(int t = 0; t < cur_span; ++t)
			for (int k = 0; k < slider_d[i]; ++k)
				lmao.push_back({j.first * dimension[p] + modulo[i] + k + stride[i] * t, j.second * cur_span + t});
		}
		perm = lmao;
	}

	std::vector<int> cnt(s2);
	for (auto& i : perm) {
		i.second = i.second * s3 + (cnt[i.second]++);
	}

	return MapValue(x, perm, std::make_pair(s1, s2 * s3), new_dimension);
}


Tensor Dropout(Tensor x, float rate) {
	std::vector<bool> cur(x.size());
	for (int i = 0; i < x.size(); ++i) 
		cur[i] = (uniform_random_float(0, 1) <= rate);
	return TensorFunction(
		[=](std::vector<float> x, int row) -> std::vector<float> {
			std::vector<float> ans(x.size());
			for (int i = 0; i < x.size(); ++i) {
				if (cur[i] == 0) ans[i] = x[i];
			}
			return ans;
		},

		[=](std::vector<float> x, int row) -> std::vector<float> {
			std::vector<float> ans(x.size());
			for (int i = 0; i < x.size(); ++i) {
				if (cur[i] == 0) ans[i] = 1;
			}
			return ans;
		}
	)(x);
}


Tensor& operator += (Tensor& x, Tensor y) { x = x + y; return x; }
Tensor& operator -= (Tensor& x, Tensor y) { x = x - y; return x; }
Tensor& operator *= (Tensor& x, Tensor y) { x = x * y; return x; }

Tensor& operator += (Tensor& x, float y) { x = x + y; return x; }
Tensor& operator -= (Tensor& x, float y) { x = x - y; return x; }
Tensor& operator *= (Tensor& x, float y) { x = x * y; return x; }
Tensor& operator /= (Tensor& x, float y) { x = x / y; return x; }