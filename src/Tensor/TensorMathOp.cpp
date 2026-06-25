#include <Tensor/TensorMathOp.hpp>
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
	float forward(float x) {return std::exp(x);}
	float backward(float x) {return std::exp(x);}
}


namespace cube_function {
	float forward(float x) {return x * x * x;}
	float backward(float x) {return 3 * x * x;}
}

namespace cbrt_function {
	float forward(float x) {return std::pow(x, 1.0 / 3);}
	float backward(float x) {
		float s = forward(x);
		return 1.0 / 3 / (s * s);
	}
}

namespace sin_function {
	float forward(float x) { return std::sin(x); }
	float backward(float x) { return std::cos(x); }
}
namespace cos_function {
	float forward(float x) { return std::cos(x); }
	float backward(float x) { return -std::sin(x); }
}
namespace tan_function {
	float forward(float x) { return std::tan(x); }
	float backward(float x) {
		float s = std::cos(x);
		return 1.0 / (s * s);
	}
}
namespace asin_function {
	float forward(float x) { return std::asin(x); }
	float backward(float x) { return 1.0 / std::sqrt(1 - x * x); }
}
namespace acos_function {
	float forward(float x) { return std::acos(x); }
	float backward(float x) { return -1.0 / std::sqrt(1 - x * x); }
}
namespace atan_function {
	float forward(float x) { return std::atan(x); }
	float backward(float x) { return 1.0 / std::sqrt(1 + x * x); }
}

#define constructFunction(NAME, FUNCTION_CLASS) \
TensorFunction NAME( \
[](std::vector<float> x, int row) -> std::vector<float> { \
	std::vector<float> ans(x.size()); \
	for (int i = 0; i < x.size(); ++i) \
		ans[i] = FUNCTION_CLASS::forward(x[i]); \
	return ans; \
	}, \
	[](std::vector<float> x, int row) -> std::vector<float> { \
	std::vector<float> ans(x.size()); \
	for (int i = 0; i < x.size(); ++i) \
		ans[i] = FUNCTION_CLASS::backward(x[i]); \
	return ans; \
	} \
);

constructFunction(reLU, reLU_function);
constructFunction(Sigmoid, sigmoid_function);
constructFunction(Abs, abs_function);
constructFunction(Sqr, sqr_function);
constructFunction(Sqrt, sqrt_function);
constructFunction(Log, log_function);
constructFunction(Tanh, tanh_function);
constructFunction(Inverse, inverse_function);
constructFunction(Exp, exp_function);
constructFunction(Cube, cube_function);
constructFunction(Cbrt, cbrt_function);
constructFunction(Sin, sin_function);
constructFunction(Cos, cos_function);
constructFunction(Tan, tan_function);
constructFunction(Asin, asin_function);
constructFunction(Acos, acos_function);
constructFunction(Atan, atan_function);


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

