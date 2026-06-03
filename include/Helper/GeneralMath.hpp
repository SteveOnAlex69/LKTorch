#ifndef GENERALMATH_HPP
#define GENERALMATH_HPP

#include <random>

unsigned long long base_random();
long long random_int(long long l, long long r);

float uniform_random_float(float l, float r);
float normal_random_float(float mean, float deviation);

float sqr(float x);


#endif