#include <GeneralMath.hpp>
#include <DebugAssist.hpp>
#include <random>
#include <math.h>

std::mt19937_64 rng(67);
unsigned long long full_bit = 0 - 1;
unsigned long long base_random() {
	return (unsigned long long) rng();
}
long long random_int(long long l, long long r) {
	return base_random() % (r - l + 1) + l;
}
float random_float(float l, float r) {
	return l + (float)base_random() / full_bit * (r - l);
}


float sqr(float x) { return x * x; }

float reLU(float x) { return (x >= 0) ? x : 0; }
float DreLU(float x) { return (x >= 0) ? 1 : 0; }

float sigmoid(float x) { return 1.0f / (1 + std::exp(-x)); }
float Dsigmoid(float x) { return sigmoid(x) * sigmoid(1 - x); }
