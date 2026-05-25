#include <iostream>
#include <LinearAlg.hpp>
int main(int argv, char* args[]) {
	std::cout << "booted" << std::endl;

	Line line_matrix(5);
	Matrix matrix(5, 6);
	for (int i = 0; i < 5; ++i) line_matrix[i] = i;
	for (int i = 0; i < 5; ++i)
		for (int j = 0; j < 6; ++j) matrix[i][j] = i * j;

	std::cout << line_matrix << " " << matrix << "\n";
	Line res_matrix = line_matrix * matrix;
	std::cout << res_matrix << "\n";

	return EXIT_SUCCESS;
}