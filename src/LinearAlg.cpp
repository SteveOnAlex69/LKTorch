#include <LinearAlg.hpp>
#include <DebugAssist.hpp>

// CONSTRUCT
Line::Line(int _n) {
	n = _n;
	a = new float[n];
	for (int i = 0; i < n; ++i) a[i] = 0;
}

// COPY
Line::Line(const Line& other) {
	n = other.size();
	a = new float[n];
	for (int i = 0; i < n; ++i) a[i] = other[i];
}

//DESTRUCT
Line::~Line() {
	release();
}

// COPY ASSIGNMENT
Line& Line::operator = (Line x) {
	release();

	n = x.size();
	a = new float[n];
	for (int i = 0; i < n; ++i) a[i] = x[i];

	return *this;
}

void Line::release() {
	n = 0;
	delete[] a;
	a = nullptr;
}


int Line::size() const {
	return n;
}

float& Line::operator [] (int x) {
	return *(a+x);
}
const float& Line::operator [] (int x) const {
	return *(a + x);
}


Line& Line::operator += (Line x) {
	if (n != x.size()) throw_error("Mismatched vector size in addition");
	for (int i = 0; i < n; ++i) a[i] += x[i];
	return *this;
}
Line& Line::operator -= (Line x) {
	if (n != x.size()) throw_error("Mismatched vector size in addition");
	for (int i = 0; i < n; ++i) a[i] -= x[i];
	return *this;
}
Line& Line::operator *= (float x) {
	for (int i = 0; i < n; ++i) a[i] *= x;
	return *this;
}
Line& Line::operator /= (float x) {
	if (x == 0) throw_error("Divided by 0 in scalar vector division");
	for (int i = 0; i < n; ++i) a[i] /= x;
	return *this;
}

Line Line::operator + (Line x) {
	Line tmp = *this;
	return tmp += x;
}
Line Line::operator - (Line x) {
	Line tmp = *this;
	return tmp -= x;
}
Line Line::operator * (float x) {
	Line tmp = *this;
	return tmp *= x;
}
Line Line::operator / (float x) {
	Line tmp = *this;
	return tmp /= x;
}

Line Line::operator * (Matrix x) {
	if (n != x.size_x()) throw_error("Mismatched matrix size in vector x matrix multiplication");
	int xm = x.size_y();
	Line res(xm);
	for (int k = 0; k < n; ++k)
		for (int j = 0; j < xm; ++j)
			res[j] += (*this)[k] * x[k][j];
	return res;
}


Matrix Line::outer_multiply(Line y) {
	Matrix ans(n, y.size());
	for (int i = 0; i < n; ++i) for (int j = 0; j < y.size(); ++j)
		ans[i][j] = a[i] * y[j];
	return ans;
}


std::ostream& operator << (std::ostream& os, Line x) {
	os << "(";
	for (int i = 0; i + 1 < x.size(); ++i) os << x[i] << ", ";
	os << x[x.size() - 1] << ")";
	return os;
}




















// CONSTRUCT
Matrix::Matrix(int _n, int _m) {
	n = _n, m = _m;
	a = new float[n * m];
	for (int i = 0; i < n * m; ++i) a[i] = 0;
}

// COPY
Matrix::Matrix(const Matrix& other) {
	n = other.size_x(), m = other.size_y();
	a = new float[n * m];
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < m; ++j)
			a[i*m+j] = other[i][j];
}

//DESTRUCT
Matrix::~Matrix() {
	release();
}

// COPY ASSIGNMENT
Matrix& Matrix::operator = (Matrix x) {
	release();

	n = x.size_x(), m = x.size_y();
	a = new float[n*m];
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < m; ++j)
			a[i * m + j] = x[i][j];

	return *this;
}

void Matrix::release() {
	n = 0;
	delete[] a;
	a = nullptr;
}

int Matrix::size_x() const {
	return n;
}
int Matrix::size_y() const {
	return m;
}

float* Matrix::operator [] (int x) {
	return (a + x * m);
}
const float* Matrix::operator [] (int x) const {
	return (a + x * m);
}

Matrix& Matrix::operator += (Matrix x) {
	if (n != x.size_x() || m != x.size_y()) throw_error("Mismatched matrix size in addition");
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < m; ++j) a[i * m + j] += x[i][j];
	return *this;
}
Matrix& Matrix::operator -= (Matrix x) {
	if (n != x.size_x() || m != x.size_y()) throw_error("Mismatched matrix size in subtraction");
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < m; ++j) a[i * m + j] -= x[i][j];
	return *this;
}
Matrix& Matrix::operator *= (float x) {
	for (int i = 0; i < n*m; ++i) a[i] *= x;
	return *this;
}
Matrix& Matrix::operator /= (float x) {
	if (x == 0) throw_error("Divided by 0 in scalar matrix division");
	for (int i = 0; i < n*m; ++i) a[i] /= x;
	return *this;
}

Matrix Matrix::operator + (Matrix x) {
	Matrix tmp = *this;
	return tmp += x;
}
Matrix Matrix::operator - (Matrix x) {
	Matrix tmp = *this;
	return tmp -= x;
}
Matrix Matrix::operator * (float x) {
	Matrix tmp = *this;
	return tmp *= x;
}
Matrix Matrix::operator / (float x) {
	Matrix tmp = *this;
	return tmp /= x;
}

Matrix Matrix::operator * (Matrix x) {
	if (m != x.size_x()) throw_error("Mismatched matrix size in mat mul");
	int xm = x.size_y();
	Matrix res(n, xm);
	for (int i = 0; i < n; ++i)
		for (int k = 0; k < m; ++k)
			for (int j = 0; j < xm; ++j)
				res[i][j] += (*this)[i][k] * x[k][j];
	return res;
}

Matrix Matrix::transpose() {
	Matrix res(m, n);
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < m; ++j)
			res[j][i] = (*this)[i][j];
	return res;
}


std::ostream& operator << (std::ostream& os, Matrix x) {
	os << "[";
	for (int i = 0; i < x.size_x(); ++i) {
		os << "(";
		for(int j = 0; j < x.size_y(); ++j) 
			os << x[i][j] << ((j + 1 == x.size_y())?"":", ");
		os << ")" << ((i + 1 == x.size_x()) ? "" : ", ");
	}
	os << "]";
	return os;
}