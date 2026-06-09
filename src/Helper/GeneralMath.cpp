#include <Helper/GeneralMath.hpp>
#include <DebugAssist/DebugAssist.hpp>
#include <random>
#include <math.h>
#include <memory>

namespace RandomDevice {
	std::mt19937_64 rng64(67);

	std::shared_ptr<std::uniform_real_distribution<float>> uniform_device;
	std::shared_ptr<std::normal_distribution<float>> normal_device;

	template <class R>
	float get_uniform_distribution(R& rng_device) {
		if (uniform_device == nullptr)
			uniform_device = std::make_shared<std::uniform_real_distribution<float>>(0.0, 1.0);
		return (*uniform_device)(rng_device);
	}

	template <class R>
	float get_normal_distribution(R& rng_device) {
		if (normal_device == nullptr)
			normal_device = std::make_shared<std::normal_distribution<float>>(0.0, 1.0);
		return (*normal_device)(rng_device);
	}
}
unsigned long long full_bit = 0 - 1;
unsigned long long base_random() {
	return (unsigned long long) RandomDevice::rng64();
}
long long random_int(long long l, long long r) {
	return base_random() % (r - l + 1) + l;
}
float uniform_random_float(float l, float r) {
	return RandomDevice::get_uniform_distribution(RandomDevice::rng64) * (r - l) + l;
}
float normal_random_float(float mean, float deviation) {
	return RandomDevice::get_normal_distribution(RandomDevice::rng64) * deviation + mean;
}
float sqr(float x) { return x * x; }