#include <Layer.hpp>
#include <math.h>

LinearLayer::LinearLayer(int n, int m) {
	W = Matrix(n, m);
	gW = Matrix(n, m);
	B = Line(m);
	gB = Line(m);
}
Line LinearLayer::forward(Line x) {
	return x * W + B;
}

Line LinearLayer::backward(Line x, Line dX) {
	gB += dX;
	gW += x.outer_multiply(dX);
	return dX * W.transpose();
}

void LinearLayer::zero_gradient() {
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < m; ++j)
			gW[i][j] = 0;
	for (int j = 0; j < m; ++j)
		gB[j] = 0;
}

