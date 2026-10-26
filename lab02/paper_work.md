# Lab 2 Paper Work: Network Design and Gradient Derivation

Student: ____________________  Student ID: ____________________  Date: __________

## 1. Network design

For hidden width `W` in `{16, 64}`:

```text
Input X (64)
   -> Linear: W1, b1
   -> ReLU: A1 (W)
   -> Linear: W2, b2
   -> ReLU: A2 (W)
   -> Linear: W3, b3
   -> Softmax: P (10)
```

Parameter shapes:

```text
W1: (64, W)    b1: (W,)
W2: (W, W)     b2: (W,)
W3: (W, 10)    b3: (10,)
```

Parameter counts:

```text
W = 16:
64*16 + 16 + 16*16 + 16 + 16*10 + 10
= 1024 + 16 + 256 + 16 + 160 + 10
= 1,482 parameters

W = 64:
64*64 + 64 + 64*64 + 64 + 64*10 + 10
= 4096 + 64 + 4096 + 64 + 640 + 10
= 8,970 parameters
```

Weights are initialized from `N(0, 0.01)` and biases are initialized to zero.

## 2. Forward propagation

```text
Z1 = X W1 + b1
A1 = ReLU(Z1)

Z2 = A1 W2 + b2
A2 = ReLU(Z2)

Z3 = A2 W3 + b3
P  = Softmax(Z3)
```

For every sample, `P` has 10 entries and:

```text
sum_k P[i,k] = 1
```

Cross-entropy loss for batch size `m`:

```text
L = -(1/m) * sum_i sum_k Y[i,k] log(P[i,k])
```

For an untrained 10-class network:

```text
L is approximately ln(10) = 2.3026
```

## 3. Backpropagation by the chain rule

Output layer:

```text
dZ3 = (P - Y) / m
dW3 = A2^T dZ3
db3 = sum_rows(dZ3)
```

Second hidden layer:

```text
dA2 = dZ3 W3^T
dZ2 = dA2 elementwise* 1[Z2 > 0]
dW2 = A1^T dZ2
db2 = sum_rows(dZ2)
```

First hidden layer:

```text
dA1 = dZ2 W2^T
dZ1 = dA1 elementwise* 1[Z1 > 0]
dW1 = X^T dZ1
db1 = sum_rows(dZ1)
```

## 4. Numerical gradient check

For one selected parameter `theta_j` and `epsilon = 1e-5`:

```text
numerical gradient
= [L(theta_j + epsilon) - L(theta_j - epsilon)] / (2*epsilon)
```

Relative error:

```text
|numerical - analytical|
-----------------------------------------------
max(1e-12, |numerical| + |analytical|)
```

Required result: relative error `< 1e-6` for at least 9 of 10 weights.

Measured result: `10/10 PASS`; largest relative error was `4.11e-7`.
