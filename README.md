## Abstract

I am trying to clone PyTorch. Obviously, my implementation suck ass, but let's try


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
- Switch out the manual matrix multiplication and shit for the actual tensor class (DONE).
- Defining linear layer, which is actually just a tensor multiplication and a tensor addition (DONE).
- Cleaning up code base by partitioning them into folders (DONE).
- Possibly speeding up the tensor operations (DONE) (by 5 times)
- Defining the Tensor Function class, which will allow people to add whatever function they want into the mix. (DONE)
	+ reLU (DONE)
	+ sigmoid (DONE)
	+ abs (DONE)
	+ log (DONE)
	+ sqrt (DONE)
	+ sqr (done)
	+ Sum (DONE)
	+ Mean (DONE)
	+ Max (DONE)
	+ Min (DONE)
	+ Scalar multiplication (DONE)
	+ Tanh (DONE)
	+ Scalar addition, subtraction (DONE)
- Defining some other basic tensor amenities:
	+ Flatten (DONE)
	+ Reshape (DONE)
	+ Transpose (DONE)
	+ Zero/One/Arbitrary Tensor (DONE)
	+ Uniform random tensor (DONE)
	+ Gaussian random tensor (DONE)
	+ Slicing the tensor (DONE)
	+ Merging the tensor (If the last n-1 dimensions match, we say OK) (DONE)
- Improving the check of the tensor operation before doing stuff. (DONE)
- Adding the "print structure" method, essential for debugging. (HALF DONE)
- Module base class (DONE)
- Adding the save and load function to a binary file
- Adding other type of optimizer, like Adam, Simulated Annealling, SGD with momentum, or Softmax.
- Sequential class.
- Adding other fun type of loss function.
- Saving and loading file
- Possibly speeding up the tensor operations
- Make a python binding