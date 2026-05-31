#ifndef DEBUGASSIST_HPP
#define DEBUGASSIST_HPP

#include <iostream>
#include <string>
#include <vector>

void throw_error(std::string s);

template <class T>
std::ostream& operator << (std::ostream& os, std::vector<T> v) {
	os << "(";
	for (int i = 0; i < (int)v.size(); ++i) {
		os << v[i];
		if (i + 1 < (int)v.size()) os << ", ";
	}
	os << ")";
	return os;
}


#endif