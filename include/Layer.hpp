#ifndef LAYER_HPP
#define LAYER_HPP

#include <DebugAssist.hpp>
#include <LinearAlg.hpp>

// this define all kinds of tensor transformation that could happen.

struct LinearLayer {
public: 
	LinearLayer(int n, int m);
	Line forward(Line x);
	Line backward(Line x, Line dX);
	void zero_gradient();

private:
	int n, m; 
	// input and output size
	
	Matrix W, gW;
	Line B, gB;
};


#endif