#include <Tensor.hpp>
#include <Layer.hpp>

Tensor::Tensor(int _n) {
	n = _n;
	A = Line(n);
}


Tensor::Tensor(Line x) {
	n = x.size();
	A = x;
}

void Tensor::set_trans_type(TransformType trans) { trans_type = trans; }
TransformType Tensor::get_trans_type() const { return trans_type; }
void Tensor::set_parent1(std::shared_ptr<Tensor> p) { parent1 = p; }
std::shared_ptr<Tensor> Tensor::get_parent1() const { return parent1; }
void Tensor::set_parent2(std::shared_ptr<Tensor> p) { parent2 = p; }
std::shared_ptr<Tensor> Tensor::get_parent2() const { return parent2; }
Line Tensor::getA() const { return A; }

void Tensor::set_linear_layer(void* l) { lilirlaler = l; }
void* Tensor::get_linear_layer() const { return lilirlaler; }



float reLU(float x) { return (x >= 0) ? x : 0; }
float DreLU(float x) { return (x >= 0) ? 1 : 0; }

float sigmoid(float x) { return 1.0f / (1 + std::exp(-x)); }
float Dsigmoid(float x) { return x * (1 - x); }


Tensor operator + (const Tensor& x, const Tensor& y) {
	Line cur1 = x.getA(), cur2 = y.getA();
	Tensor ans(cur1 + cur2);
	ans.set_trans_type(ADD);
	ans.set_parent1(std::make_shared<Tensor>(x));
	return ans;
}


Tensor operator - (const Tensor& x, const Tensor& y) {
	Line cur1 = x.getA(), cur2 = y.getA();
	Tensor ans(cur1 - cur2);
	ans.set_trans_type(SUBTRACT);
	ans.set_parent1(std::make_shared<Tensor>(x));
	return ans;
}


void Tensor::backward(Line dL) {
	if (trans_type == ADD) {
		parent1->backward(dL);
		parent2->backward(dL);
	}
	if (trans_type == SUBTRACT) {
		parent1->backward(dL);
		parent2->backward(dL * (-1));
	}
	if (trans_type == LINEAR) {
		Line X = parent1->getA(), dX = dL;
		dL = ((LinearLayer*)lilirlaler)->backward(X, dX);
		parent1->backward(dL);
	}
	if (trans_type == RELU) {
		for (int i = 0; i < dL.size(); ++i)
			dL[i] *= DreLU(dL[i]);
		parent1->backward(dL);
	}
	if (trans_type == SIGMOID) {
		for (int i = 0; i < dL.size(); ++i)
			dL[i] *= Dsigmoid(dL[i]);
		parent1->backward(dL);
	}
}

Tensor Tensor::forward(LinearLayer &ll) {
	Line cur = ll.forward(getA());
	Tensor ans(cur);
	ans.set_trans_type(LINEAR);
	ans.set_linear_layer((void*) ( & ll));
	ans.set_parent1(std::make_shared<Tensor>(*this));
	return ans;
}


Tensor reLU(const Tensor& x) {
	Line cur = x.getA();
	for (int i = 0; i < cur.size(); ++i)
		cur[i] = reLU(cur[i]);
	Tensor ans(cur);
	ans.set_trans_type(RELU);
	ans.set_parent1(std::make_shared<Tensor>(x));
	return ans;
}
Tensor sigmoid(const Tensor& x) {
	Line cur = x.getA();
	for (int i = 0; i < cur.size(); ++i)
		cur[i] = sigmoid(cur[i]);
	Tensor ans(cur);
	ans.set_trans_type(SIGMOID);
	ans.set_parent1(std::make_shared<Tensor>(x));
	return ans;
}