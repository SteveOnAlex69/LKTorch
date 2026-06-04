#include <Tensor/TensorOperation.hpp>
#include <Tensor/TensorInit.hpp>
#include <DebugAssist/DebugAssist.hpp>
#include <Tensor/DebugTensor.hpp>



namespace reLU_function {
	float forward(float x) { return (x >= 0) ? x : 0; }
	float backward(float x) { return (x >= 0) ? 1 : 0; }
}

namespace sigmoid_function {
	float forward(float x) { return 1.0f / (1 + std::exp(-x)); }
	float backward(float x) { return forward(x) * forward(1 - x); }
}

namespace abs_function {
	float forward(float x) { return (x >= 0) ? x : -x; }
	float backward(float x) { return (x >= 0) ? 1 : -1; }
}

namespace sqr_function {
	float forward(float x) { return x * x; }
	float backward(float x) { return x * 2; }
}

namespace sqrt_function {
	float forward(float x) { return sqrt(x); }
	float backward(float x) { return 0.5 / sqrt(x); }
}

namespace log_function {
	float forward(float x) { return log(x); }
	float backward(float x) { return 1 / x; }
}

namespace tanh_function {
	float forward(float x) { return 2 / (1 + std::exp(-2 * x)) - 1; }
	float backward(float x) { return 1 - sqr_function::forward(forward(x)); }
}
namespace inverse_function {
	float forward(float x) { return 1 / x; }
	float backward(float x) { return -1 / (x * x); }
}



TensorFunction reLU(
	[&](StaticFloatVector& x) -> StaticFloatVector {
		StaticFloatVector ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = reLU_function::forward(x[i]);
		return ans;
	},

	[&](StaticFloatVector& x) -> StaticFloatVector {
		StaticFloatVector ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = reLU_function::backward(x[i]);
		return ans;
	}
);
TensorFunction Sigmoid(
	[&](StaticFloatVector& x) -> StaticFloatVector {
		StaticFloatVector ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = sigmoid_function::forward(x[i]);
		return ans;
	},

	[&](StaticFloatVector& x) -> StaticFloatVector {
		StaticFloatVector ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = sigmoid_function::backward(x[i]);
		return ans;
	}
);
TensorFunction Abs(
	[&](StaticFloatVector& x) -> StaticFloatVector {
		StaticFloatVector ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = abs_function::forward(x[i]);
		return ans;
	},

	[&](StaticFloatVector& x) -> StaticFloatVector {
		StaticFloatVector ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = abs_function::backward(x[i]);
		return ans;
	}
);
TensorFunction Sqr(
	[&](StaticFloatVector& x) -> StaticFloatVector {
		StaticFloatVector ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = sqr_function::forward(x[i]);
		return ans;
	},

	[&](StaticFloatVector& x) -> StaticFloatVector {
		StaticFloatVector ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = sqr_function::backward(x[i]);
		return ans;
	}
);
TensorFunction Sqrt(
	[&](StaticFloatVector& x) -> StaticFloatVector {
		StaticFloatVector ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = sqrt_function::forward(x[i]);
		return ans;
	},

	[&](StaticFloatVector& x) -> StaticFloatVector {
		StaticFloatVector ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = sqrt_function::backward(x[i]);
		return ans;
	}
);
TensorFunction Log(
	[&](StaticFloatVector& x) -> StaticFloatVector {
		StaticFloatVector ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = log_function::forward(x[i]);
		return ans;
	},

	[&](StaticFloatVector& x) -> StaticFloatVector {
		StaticFloatVector ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = log_function::backward(x[i]);
		return ans;
	}
);

TensorFunction Tanh(
	[&](StaticFloatVector& x) -> StaticFloatVector {
		StaticFloatVector ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = tanh_function::forward(x[i]);
		return ans;
	},

	[&](StaticFloatVector& x) -> StaticFloatVector {
		StaticFloatVector ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = tanh_function::backward(x[i]);
		return ans;
	}
);


TensorFunction Inverse(
	[&](StaticFloatVector& x) -> StaticFloatVector {
		StaticFloatVector ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = inverse_function::forward(x[i]);
		return ans;
	},

	[&](StaticFloatVector& x) -> StaticFloatVector {
		StaticFloatVector ans(x.size());
		for (int i = 0; i < x.size(); ++i)
			ans[i] = inverse_function::backward(x[i]);
		return ans;
	}
);

TensorFunction Max(
	[&](StaticFloatVector& x) -> StaticFloatVector {
		StaticFloatVector ans(x.size());
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

	[&](StaticFloatVector& x) -> StaticFloatVector {
		StaticFloatVector ans(x.size());
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
	[&](StaticFloatVector& x) -> StaticFloatVector {
		StaticFloatVector ans(x.size());
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

	[&](StaticFloatVector& x) -> StaticFloatVector {
		StaticFloatVector ans(x.size());
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

Tensor Tensor::Transpose() { return Tensor(transpose(ts)); }
Tensor Tensor::Reshape(StaticIntVector y) { return Tensor(reshape(ts, y)); }
Tensor Tensor::Slice(StaticIntVector l, StaticIntVector r) { return Tensor(slice(ts, l, r)); }
Tensor Tensor::Merge(Tensor y) { return Tensor(merge(ts, y.ts)); }

Tensor Sum(Tensor x) {
	int x_size = x.get_tensor_size();
	Tensor tmp = x.Reshape(std::vector<int> {1, x_size}) * Tensor(std::vector<int>{x_size, 1}, std::vector<float>(x_size, 1));
	return tmp.Reshape(std::vector<int>(0));
}

Tensor Flatten(Tensor x) {
	return x.Reshape(std::vector<int>{x.get_tensor_size()});
}

Tensor operator * (Tensor x, float y) {
	StaticIntVector d = x.get_tensor_dimension();
	StaticIntVector d1(d.size() + 1);
	for (int i = 0; i < d.size(); ++i) d1[i] = d[i];
	d1[d.size()] = 1;
	return x.Reshape(d1) * UniformValue(std::vector<int>{1}, y);
}


Tensor operator / (Tensor x, float y) {
	StaticIntVector d = x.get_tensor_dimension();
	StaticIntVector d1(d.size() + 1);
	for (int i = 0; i < d.size(); ++i) d1[i] = d[i];
	d1[d.size()] = 1;
	return x.Reshape(d1) * UniformValue(std::vector<int>{1}, 1/y);
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

Tensor Mean(Tensor x) {
	return Sum(x) / float(x.get_tensor_size());
}
Tensor SoftMax(Tensor x) {
	StaticIntVector d = x.get_tensor_dimension();
	StaticIntVector d1(d.size() + 1);
	for (int i = 0; i < d.size(); ++i) d1[i] = d[i];
	d1[d.size()] = 1;
	return x.Reshape(d1) * Inverse(Sum(x)).Reshape(std::vector<int>{1});
}


Tensor Huber(Tensor x, float epsilon) {
	TensorFunction Goober(
		[&](StaticFloatVector& x) -> StaticFloatVector {
			StaticFloatVector ans(x.size());
			for (int i = 0; i < x.size(); ++i) {
				if (abs(x[i]) >= epsilon) ans[i] = abs(x[i]) * epsilon - 0.5 * sqr_function::forward(x[i]);
				else ans[i] = 0.5 * sqr_function::forward(x[i]);
			}
			return ans;
		},

		[&](StaticFloatVector& x) -> StaticFloatVector {
			StaticFloatVector ans(x.size());
			for (int i = 0; i < x.size(); ++i) {
				if (abs(x[i]) >= epsilon) 
					ans[i] = abs_function::backward(x[i]) * epsilon - 0.5 * sqr_function::backward(x[i]);
				else 
					ans[i] = 0.5 * sqr_function::backward(x[i]);
			}
			return ans;
		}
	);
	return Goober(x);
}