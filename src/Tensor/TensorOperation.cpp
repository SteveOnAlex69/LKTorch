#include <Tensor/TensorOperation.hpp>



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

Tensor Sum(Tensor x) {
	int x_size = x.get_tensor_size();
	Tensor tmp = x.ChangeDimension(std::vector<int> {1, x_size}) * Tensor(std::vector<int>{x_size, 1}, std::vector<float>(x_size, 1));
	return ChangeDimension(tmp, std::vector<int>(0));
}

Tensor Flatten(Tensor x) {
	return x.ChangeDimension(std::vector<int>{x.get_tensor_size()});
}

Tensor operator * (Tensor x, float y) {
	TensorFunction Multiply(
		[&](StaticFloatVector& x) -> StaticFloatVector {
			StaticFloatVector ans(x.size());
			for (int i = 0; i < x.size(); ++i)
				ans[i] = x[i] * y;
			return ans;
		},

		[&](StaticFloatVector& x) -> StaticFloatVector {
			StaticFloatVector ans(x.size());
			for (int i = 0; i < x.size(); ++i)
				ans[i] = x[i] / y;
			return ans;
		}
	);
	return Multiply(x);
}


Tensor operator / (Tensor x, float y) {
	TensorFunction Division(
		[&](StaticFloatVector& x) -> StaticFloatVector {
			StaticFloatVector ans(x.size());
			for (int i = 0; i < x.size(); ++i)
				ans[i] = x[i] / y;
			return ans;
		},

		[&](StaticFloatVector& x) -> StaticFloatVector {
			StaticFloatVector ans(x.size());
			for (int i = 0; i < x.size(); ++i)
				ans[i] = x[i] * y;
			return ans;
		}
	);
	return Division(x);
}

Tensor mean(Tensor x) {
	return Sum(x) / float(x.get_tensor_size());
}