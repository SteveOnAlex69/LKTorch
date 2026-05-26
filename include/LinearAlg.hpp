#ifndef LINEARALG_HPP
#define LINEARALG_HPP

#include <iostream>

class Line;
class Matrix;

class Line {
public:
	Line(int _n = 0);
	Line(const Line& other);
	~Line();
	
	int size() const;
	
	float& operator [] (int x);
	const float& operator [] (int x) const;
	Line& operator = (Line x);
	Line& operator += (Line x);
	Line& operator -= (Line x);
	Line& operator *= (float x);
	Line& operator /= (float x);

	Line operator + (Line x);
	Line operator - (Line x);
	Line operator * (float x);
	Line operator / (float x);

	Line operator * (Matrix x);

	Matrix outer_multiply(Line y);
private:
	int n;
	float* a;

	void release();
};

std::ostream& operator << (std::ostream& os, Line x);

class Matrix {
public:
	Matrix(int _n = 0, int _m = 0);
	Matrix(const Matrix& other);
	~Matrix();

	int size_x() const;
	int size_y() const;

	float* operator [] (int x);
	const float* operator [] (int x) const;

	Matrix& operator = (Matrix x);
	Matrix& operator += (Matrix x);
	Matrix& operator -= (Matrix x);
	Matrix& operator *= (float x);
	Matrix& operator /= (float x);

	Matrix operator + (Matrix x);
	Matrix operator - (Matrix x);
	Matrix operator * (float x);
	Matrix operator / (float x);
	Matrix operator * (Matrix x);


	Matrix transpose();
private:
	int n, m;
	float* a;

	void release();
};

std::ostream& operator << (std::ostream& os, Matrix x);

#endif