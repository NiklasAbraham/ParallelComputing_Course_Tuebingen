# Exercise 01: Regular Data Parallelism

## N-Body Gravitational Simulation

Simulate the gravitational interactions between N particles in 2D space. Each
particle has a two-dimensional position `p` and velocity `v`, and the same
mass `1`. The gravitational constant is `1` and the time step is `0.01`.

Compute the gravitational force `f` on each particle from all other particles
and the resulting acceleration `a`. Add the acceleration to the velocity and the
velocity to the position and return both. Be careful to mask the force each
particle would exert onto itself.

Use file `nbody.py` as a starting point and try to stick to what we have covered
in the lecture: operations, reductions, broadcasting, etc.

## Convolutional Neural Network

Apply a 2D convolution with ReLU activation to a small grayscale (float32)
image, then classify it into one of 10 digit classes (0-9). The network has the
following components:

- Convolution layer: applies the kernel to the image, then adds a bias. The
  kernel has shape [3, 3] and slides over the input image [16, 16] to produce a
  feature map.

- ReLU activation: applies element-wise maximum with zero.

- Flatten: converts the 2D feature map into a 1D vector.

- Linear layer: multiplies the flattened vector by a weight matrix [10, 196] and
  adds a bias [10] to produce 10 output logits.

The output is a vector of 10 logits, one per digit class.

Use file `cnn.py` as a starting point and try to stick to what we have covered
in the lecture: operations, broadcasting, unfold, etc.

## Multi-Head Self-Attention

Extend the given single-head attention implementation to multi-head attention
with H heads running in parallel. Instead of computing one attention output of
dimension [N, D], compute H independent attention heads simultaneously. Each
head has its own query_weights, key_weights, and value_weights matrices.

input: [N, M] - sequence of N tokens, each M-dimensional
query_weights: [H, M, D] - query weights for H heads
key_weights: [H, M, D] - key weights for H heads
value_weights: [H, M, D] - value weights for H heads

output: [H, N, D] - attention output for each of the H heads

Each head computes attention independently using the same formula as before, but
now you need to handle the batch of H heads in parallel using broadcasting etc.

Use file `attention.py` as a starting point and extend the single-head attention
implementation. Try to stick to what we have covered in the lecture.



## Niklas Notes

Commands from the lecture:

| Concept              | Command/Function                        | Description                                                                                                  |
|----------------------|-----------------------------------------|--------------------------------------------------------------------------------------------------------------|
| Tensor Creation      | `torch.tensor()`                        | Creates a multi-dimensional array (tensor) in PyTorch.                                                       |
| Element-wise Op      | `a + b`, `$2^{*}a$`                     | Performs parallel, element-wise operations, exploiting SIMD parallelism.                                     |
| Shape Manipulation   | `torch.unsqueeze(input, dim)`           | Returns a new tensor with a dimension of size one inserted at the specified position.                        |
| Reduction            | `x.sum(dim=...)`                        | Aggregates tensor values (e.g., summation) along specified dimensions, using parallel tree-shaped reduction. |
| Softmax              | `torch.exp()`, `x.max()`                | Vectorized components of the Softmax function, enabling parallel computation of the exponential and maximum. |
| Masking/Condition    | `x > 0` (Mask), `torch.where(mask, x, 0)`| Emulates control flow by creating a boolean mask and conditionally selecting elements or zeroing them out.   |
| Convolution Setup    | `input.unfold(dimension, size, step)`   | Returns a view of the tensor containing all overlapping slices (windows) necessary for parallel convolution. |
| Irregular Gather     | `x[i]`                                  | Selects elements from tensor x using indices specified in tensor i.                                          |
| Irregular Scatter    | `r.scatter_add(0, i, x)`                | Accumulates (adds) values from source tensor r based on indices i.                                          |