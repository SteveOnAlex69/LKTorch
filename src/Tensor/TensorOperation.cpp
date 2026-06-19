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
	float forward(float x) { return 1.0f / (1 + std::exp(-x)); }
	float backward(float x) { return forward(x) * forward(1 - x); }
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
	float forward(float x) { return 2 / (1 + std::exp(-2 * x)) - 1; }
	float backward(float x) { return 1 - sqr_function::forward(forward(x)); }
}

namespace inverse_function {
	const float EPS = 1e-5;
	const float SEPS = 1 / EPS;
	float forward(float x) { 
		if (x >= EPS) return 1 / x; 
		return SEPS - (x - EPS) * (1 / (EPS * EPS));
	}
	float backward(float x) { 
		if (x >= EPS) return -1 / (x * x);
		return -1 / (EPS * EPS);
	}
}



TensorFunction reLU(
	[](std::vector<float> x) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = reLU_function::forward(x[i]);
		return ans;
	},

	[](std::vector<float> x) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = reLU_function::backward(x[i]);
		return ans;
	}
);
TensorFunction Sigmoid(
	[](std::vector<float> x) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = sigmoid_function::forward(x[i]);
		return ans;
	},

	[](std::vector<float> x) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = sigmoid_function::backward(x[i]);
		return ans;
	}
);
TensorFunction Abs(
	[](std::vector<float> x) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = abs_function::forward(x[i]);
		return ans;
	},

	[](std::vector<float> x) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = abs_function::backward(x[i]);
		return ans;
	}
);
TensorFunction Sqr(
	[](std::vector<float> x) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = sqr_function::forward(x[i]);
		return ans;
	},

	[](std::vector<float> x) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = sqr_function::backward(x[i]);
		return ans;
	}
);
TensorFunction Sqrt(
	[](std::vector<float> x) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = sqrt_function::forward(x[i]);
		return ans;
	},

	[](std::vector<float> x) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = sqrt_function::backward(x[i]);
		return ans;
	}
);
TensorFunction Log(
	[](std::vector<float> x) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = log_function::forward(x[i]);
		return ans;
	},

	[](std::vector<float> x) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = log_function::backward(x[i]);
		return ans;
	}
);

TensorFunction Tanh(
	[](std::vector<float> x) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = tanh_function::forward(x[i]);
		return ans;
	},

	[](std::vector<float> x) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = tanh_function::backward(x[i]);
		return ans;
	}
);


TensorFunction Inverse(
	[](std::vector<float> x) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = inverse_function::forward(x[i]);
		return ans;
	},

	[](std::vector<float> x) -> std::vector<float> {
		std::vector<float> ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = inverse_function::backward(x[i]);
		return ans;
	}
);

TensorFunction Max(
	[](std::vector<float> x) -> std::vector<float> {
		std::vector<float> ans(x.size());
		float max = -1e18;
		int idx = 0;
		for (int i = 0; i < x.size(); ++i) 
			if (max < x[i]) {
				max = x[i];
				idx = i;
			}
		ans[idx] = max;
		return ans;
	},

	[](std::vector<float> x) -> std::vector<float> {
		std::vector<float> ans(x.size());
		float max = -1e18;
		int idx = 0;
		for (int i = 0; i < x.size(); ++i)
			if (max < x[i]) {
				max = x[i];
				idx = i;
			}
		ans[idx] = 1;
		return ans;
	}
);


TensorFunction Min(
	[](std::vector<float> x) -> std::vector<float> {
		std::vector<float> ans(x.size());
		float min = 1e18;
		int idx = 0;
		for (int i = 0; i < x.size(); ++i)
			if (min > x[i]) {
				min = x[i];
				idx = i;
			}
		ans[idx] = min;
		return ans;
	},

	[](std::vector<float> x) -> std::vector<float> {
		std::vector<float> ans(x.size());
		float min = 1e18;
		int idx = 0;
		for (int i = 0; i < x.size(); ++i)
			if (min > x[i]) {
				min = x[i];
				idx = i;
			}
		ans[idx] = 1;
		return ans;
	}
);


Tensor Permute(Tensor x, std::vector<int> new_d, std::vector<int> new_p) { return x.Permute(new_d, new_p); }
Tensor PermuteDimension(Tensor x, std::vector<int> perm) { return x.PermuteDimension(perm); }
Tensor Transpose(Tensor x) { return x.Transpose();; }
Tensor Reshape(Tensor x, std::vector<int> y) { return x.Reshape(y); }
Tensor Slice(Tensor x, std::vector<int> l, std::vector<int> r) { return x.Slice(l, r); }
Tensor Merge(Tensor x, Tensor y) { return x.Merge(y); }
Tensor ValueMultiply(Tensor x, Tensor y) { return x.ValueMultiply(y); }


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

Tensor Flatten(Tensor x) {
	return x.Reshape(std::vector<int>{x.size()});
}


Tensor SumAll(Tensor x) { return Sum(Flatten(x)); }
Tensor MeanAll(Tensor x) { return Mean(Flatten(x)); }



Tensor operator * (Tensor x, float y) {
	return TensorFunction(
		std::function<std::vector<float>(std::vector<float>)>([=](std::vector<float> x) -> std::vector<float> {
			std::vector<float> ans(x.size());
			for (int i = 0; i < x.size(); ++i)
				ans[i] = x[i] * y;
			return ans;
		}),

		std::function<std::vector<float>(std::vector<float>)>([=](std::vector<float> x) -> std::vector<float> {
			std::vector<float> ans(x.size());
			for (int i = 0; i < x.size(); ++i)
				ans[i] = y;
			return ans;
		})
	)(x);
}
Tensor operator / (Tensor x, float y) {
	return TensorFunction(
		std::function<std::vector<float>(std::vector<float>)>([=](std::vector<float> x) -> std::vector<float> {
			std::vector<float> ans(x.size());
			for (int i = 0; i < x.size(); ++i)
				ans[i] = x[i] / y;
			return ans;
			}),

		std::function<std::vector<float>(std::vector<float>)>([=](std::vector<float> x) -> std::vector<float> {
			std::vector<float> ans(x.size());
			for (int i = 0; i < x.size(); ++i)
				ans[i] = 1.0/y;
			return ans;
			})
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
	int x_size = x.size(), d2 = d[d.size() - 1];
	x = x.Reshape(std::vector<int>{x_size / d2, d2}); y = Flatten(y);
	x = Transpose(x);
	x = x.ValueMultiply(y);
	x = Transpose(x);
	x = x.Reshape(d);
	return x;
}


Tensor Huber(Tensor x, float epsilon) {
	return TensorFunction(
		[=](std::vector<float> x) -> std::vector<float> {
			std::vector<float> ans(x.size());
			for (int i = 0; i < x.size(); ++i) {
				if (abs(x[i]) >= epsilon) ans[i] = abs(x[i]) * epsilon - 0.5 * epsilon * epsilon;
				else ans[i] = 0.5 * sqr_function::forward(x[i]);
			}
			return ans;
		},

		[=](std::vector<float> x) -> std::vector<float> {
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


Tensor Unfold(Tensor x, int a, int b) {
	StaticIntVector d = x.dimension();
	int n = d[d.size() - 2], m = d[d.size() - 1];
	x = x.Slice(std::vector<int>{0, 0}, std::vector<int>{n - n % a-1, m - m % b-1});
	x = x.Reshape(std::vector<int>{n / 3, 3, m / 3, 3});
	x = x.PermuteDimension({0, 2, 1, 3});
	return x;
}


Tensor& operator += (Tensor& x, Tensor y) { x = x + y; return x; }
Tensor& operator -= (Tensor& x, Tensor y) { x = x - y; return x; }
Tensor& operator *= (Tensor& x, Tensor y) { x = x * y; return x; }

Tensor& operator += (Tensor& x, float y) { x = x + y; return x; }
Tensor& operator -= (Tensor& x, float y) { x = x - y; return x; }
Tensor& operator *= (Tensor& x, float y) { x = x * y; return x; }
Tensor& operator /= (Tensor& x, float y) { x = x / y; return x; }