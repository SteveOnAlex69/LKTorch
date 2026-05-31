## Todo list

- Define a vector, which is just a one dimensional array. Define addition, subtraction, scalar multiplication. (DONE)
- Define a matrix, which is a two dimentional array. Define addition, subtraction, scalar multiplication, and multiplication. (DONE)
- Define a transformation, which stores: (DONE)
	+ The type of transformation it is (activation function, or matrix computation). If it's matrix computation:
		* The current matrix weight (two matrices, w and b).
		* The current gradient (two matrices, gw and gb).
- Define a tensor, which stores: (DONE)
	+ The current vector
	+ The previous state, and which kind of transformation gets it there.
- Define an optimizer class, and we will just use the basic SGD for this one. (DONE)

- Try to train the AI on the function f(x, y) = e^(-x^2-y^2) (DONE)
- Swap 