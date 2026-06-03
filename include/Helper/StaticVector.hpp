#ifndef STATICVECTOR_HPP
#define STATICVECTOR_HPP

#include <vector>
#include <iostream>

// adding template is easy, I'll just stick with float for now
class StaticFloatVector {
public:
	StaticFloatVector(int _n = 0);
	StaticFloatVector(const StaticFloatVector& other);
	StaticFloatVector(std::vector<float> other);

	~StaticFloatVector();

	int size() const;


	inline float& operator [] (int x) { return *(a + x); }
	inline const float& operator [] (int x) const { return *(a + x); }
	StaticFloatVector& operator = (StaticFloatVector x);

	void clear();

	float* data() { return a; }
	const float* data() const { return a; }
private:
	int n;
	float* a;
	void release();
};
StaticFloatVector& operator += (StaticFloatVector& a, StaticFloatVector b);
StaticFloatVector& operator -= (StaticFloatVector& a, StaticFloatVector b);
StaticFloatVector& operator *= (StaticFloatVector& a, float x);


StaticFloatVector operator + (StaticFloatVector a, StaticFloatVector b);
StaticFloatVector operator - (StaticFloatVector a, StaticFloatVector b);
StaticFloatVector operator * (StaticFloatVector a, float x);

bool operator == (StaticFloatVector a, StaticFloatVector b);
bool operator != (StaticFloatVector a, StaticFloatVector b);


class StaticIntVector {
public:
	StaticIntVector(int _n = 0);
	StaticIntVector(const StaticIntVector& other);
	StaticIntVector(std::vector<int> other);
	~StaticIntVector();

	int size() const;


	inline int& operator [] (int x) { return *(a + x); }
	inline const int& operator [] (int x) const { return *(a + x); }
	StaticIntVector& operator = (StaticIntVector x);

	void clear();
private:
	int n;
	int* a;
	void release();
};


StaticIntVector& operator += (StaticIntVector& a, StaticIntVector b);
StaticIntVector& operator -= (StaticIntVector& a, StaticIntVector b);
StaticIntVector& operator *= (StaticIntVector& a, float x);


StaticIntVector operator + (StaticIntVector a, StaticIntVector b);
StaticIntVector operator - (StaticIntVector a, StaticIntVector b);
StaticIntVector operator * (StaticIntVector a, float x);

bool operator == (StaticIntVector a, StaticIntVector b);
bool operator != (StaticIntVector a, StaticIntVector b);

std::ostream& operator << (std::ostream& os, StaticFloatVector v);
std::ostream& operator << (std::ostream& os, StaticIntVector v);

#endif