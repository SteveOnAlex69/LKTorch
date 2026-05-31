#include <StaticVector.hpp>
#include <DebugAssist.hpp>


// CONSTRUCT
StaticFloatVector::StaticFloatVector(int _n) {
	n = _n;
	a = new float[n];
	for (int i = 0; i < n; ++i) a[i] = 0;
}

// COPY
StaticFloatVector::StaticFloatVector(const StaticFloatVector& other) {
	n = other.size();
	a = new float[n];
	for (int i = 0; i < n; ++i) a[i] = other[i];
}


StaticFloatVector::StaticFloatVector(std::vector<float> other) {
	n = other.size();
	a = new float[n];
	for (int i = 0; i < n; ++i) a[i] = other[i];
}

//DESTRUCT
StaticFloatVector::~StaticFloatVector() {
	release();
}

// COPY ASSIGNMENT
StaticFloatVector& StaticFloatVector::operator = (StaticFloatVector x) {
	release();

	n = x.size();
	a = new float[n];
	for (int i = 0; i < n; ++i) a[i] = x[i];

	return *this;
}

void StaticFloatVector::release() {
	n = 0;
	delete[] a;
	a = nullptr;
}

void StaticFloatVector::clear() { release(); }

int StaticFloatVector::size() const {
	return n;
}

float& StaticFloatVector::operator [] (int x) {
	return *(a + x);
}
const float& StaticFloatVector::operator [] (int x) const {
	return *(a + x);
}


StaticFloatVector& operator += (StaticFloatVector& a, StaticFloatVector b) {
	if (a.size() != b.size()) throw_error("Mismatched size in static float vector addition");
	int n = a.size();
	for (int i = 0; i < n; ++i) a[i] += b[i];
	return a;
}
StaticFloatVector& operator -= (StaticFloatVector& a, StaticFloatVector b) {
	if (a.size() != b.size()) throw_error("Mismatched size in static float vector subtraction");
	int n = a.size();
	for (int i = 0; i < n; ++i) a[i] -= b[i];
	return a;
}
StaticFloatVector& operator *= (StaticFloatVector& a, float x) {
	int n = a.size();
	for (int i = 0; i < n; ++i) a[i] *= x;
	return a;
}


StaticFloatVector operator + (StaticFloatVector a, StaticFloatVector b) { return a += b; }
StaticFloatVector operator - (StaticFloatVector a, StaticFloatVector b) { return a -= b; }
StaticFloatVector operator * (StaticFloatVector a, float x) { return a *= x; }


bool operator == (StaticFloatVector a, StaticFloatVector b) {
	if (a.size() != b.size()) return false;
	for (int i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
	return true;
}
bool operator != (StaticFloatVector a, StaticFloatVector b) {
	if (a.size() != b.size()) return true;
	for (int i = 0; i < a.size(); ++i) if (a[i] != b[i]) return true;
	return false;
}







// CONSTRUCT
StaticIntVector::StaticIntVector(int _n) {
	n = _n;
	a = new int[n];
	for (int i = 0; i < n; ++i) a[i] = 0;
}

// COPY
StaticIntVector::StaticIntVector(const StaticIntVector& other) {
	n = other.size();
	a = new int[n];
	for (int i = 0; i < n; ++i) a[i] = other[i];
}


StaticIntVector::StaticIntVector(std::vector<int> other) {
	n = other.size();
	a = new int[n];
	for (int i = 0; i < n; ++i) a[i] = other[i];
}

//DESTRUCT
StaticIntVector::~StaticIntVector() {
	release();
}

// COPY ASSIGNMENT
StaticIntVector& StaticIntVector::operator = (StaticIntVector x) {
	release();

	n = x.size();
	a = new int[n];
	for (int i = 0; i < n; ++i) a[i] = x[i];

	return *this;
}

void StaticIntVector::release() {
	n = 0;
	delete[] a;
	a = nullptr;
}

void StaticIntVector::clear() { release(); }

int StaticIntVector::size() const {
	return n;
}

int& StaticIntVector::operator [] (int x) {
	return *(a + x);
}
const int& StaticIntVector::operator [] (int x) const {
	return *(a + x);
}


StaticIntVector& operator += (StaticIntVector& a, StaticIntVector b) {
	if (a.size() != b.size()) throw_error("Mismatched size in static float vector addition");
	int n = a.size();
	for (int i = 0; i < n; ++i) a[i] += b[i];
	return a;
}
StaticIntVector& operator -= (StaticIntVector& a, StaticIntVector b) {
	if (a.size() != b.size()) throw_error("Mismatched size in static float vector subtraction");
	int n = a.size();
	for (int i = 0; i < n; ++i) a[i] -= b[i];
	return a;
}
StaticIntVector& operator *= (StaticIntVector& a, float x) {
	int n = a.size();
	for (int i = 0; i < n; ++i) a[i] *= x;
	return a;
}

StaticIntVector operator + (StaticIntVector a, StaticIntVector b) { return a += b; }
StaticIntVector operator - (StaticIntVector a, StaticIntVector b) { return a -= b; }
StaticIntVector operator * (StaticIntVector a, float x) { return a *= x; }


bool operator == (StaticIntVector a, StaticIntVector b) {
	if (a.size() != b.size()) return false;
	for (int i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
	return true;
}
bool operator != (StaticIntVector a, StaticIntVector b) {
	if (a.size() != b.size()) return true;
	for (int i = 0; i < a.size(); ++i) if (a[i] != b[i]) return true;
	return false;
}


std::ostream& operator << (std::ostream& os, StaticFloatVector v) {
	os << "(";
	for (int i = 0; i < v.size(); ++i) {
		os << v[i];
		if (i + 1 < v.size()) os << ", ";
	}
	os << ")";
	return os;
}
std::ostream& operator << (std::ostream& os, StaticIntVector v) {
	os << "(";
	for (int i = 0; i < v.size(); ++i) {
		os << v[i];
		if (i + 1 < v.size()) os << ", ";
	}
	os << ")";
	return os;
}