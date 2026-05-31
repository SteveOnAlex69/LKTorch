#ifndef GENERALMATH_HPP
#define GENERALMATH_HPP

unsigned long long base_random();
long long random_int(long long l, long long r);
float random_float(float l, float r);

float sqr(float x);

float reLU(float x);
float DreLU(float x);

float sigmoid(float x);
float Dsigmoid(float x);

#endif