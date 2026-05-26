## Todo list

- Define a vector, which is just a one dimensional array. Define addition, subtraction, scalar multiplication. (DONE)
- Define a matrix, which is a two dimentional array. Define addition, subtraction, scalar multiplication, and multiplication. (DONE)
- Define a transformation, which stores:
	+ The type of transformation it is (activation function, or matrix computation). If it's matrix computation:
		* The current matrix weight (two matrices, w and b).
		* The current gradient (two matrices, gw and gb).
- Define a tensor, which stores:
	+ The current vector
	+ The previous state, and which kind of transformation gets it there.
- Define an optimizer class, and we will just use the basic SGD for this one.