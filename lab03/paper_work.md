# Lab 3 Paper Work: Parameters and Representative FLOPs

Student: ____________________  Student ID: ____________________  Date: __________

## 1. Parameter calculations

### MLP

```text
Linear 784 -> 256: 784*256 + 256 = 200,960
Linear 256 -> 128: 256*128 + 128 = 32,896
Linear 128 -> 10:  128*10 + 10    = 1,290

Total = 200,960 + 32,896 + 1,290
      = 235,146 parameters
```

### CNN

```text
Conv2d 1 -> 16:
16*(1*3*3) + 16 = 160

Conv2d 16 -> 32:
32*(16*3*3) + 32 = 4,640

Linear 32*7*7 -> 10:
1,568*10 + 10 = 15,690

Total = 160 + 4,640 + 15,690
      = 20,490 parameters
```

### Attention

```text
Patch projection 49 -> 64:
49*64 + 64 = 3,200

Learned positional embedding:
16*64 = 1,024

One Transformer encoder layer:
Multi-head attention = 16,640
Feedforward network = 16,576
Two LayerNorms      =    256
Layer total         = 33,472

Two encoder layers:
2*33,472 = 66,944

Classifier 64 -> 10:
64*10 + 10 = 650

Total = 3,200 + 1,024 + 66,944 + 650
      = 71,818 parameters
```

The exact architectures given in the manual do not meet its stated 100k-150k target: MLP is above it, while CNN and Attention are below it.

## 2. Representative FLOPs

The convention counts one multiplication and one addition as two FLOPs.

### Linear layer: first MLP layer

```text
FLOPs = 2 * input * output
      = 2 * 784 * 256
      = 401,408 FLOPs
```

### Conv2d layer: first CNN convolution

```text
FLOPs = 2 * K * K * C_in * C_out * H_out * W_out
      = 2 * 3 * 3 * 1 * 16 * 28 * 28
      = 225,792 FLOPs
```

### One Transformer encoder layer

Use `sequence length = 16`, `d_model = 64`, and `d_ff = 128`.

```text
QKV projections:
2 * 16 * 64 * 3 * 64 = 393,216

Attention scores, Q K^T:
2 * 16^2 * 64 = 32,768

Weighted values, Attention V:
2 * 16^2 * 64 = 32,768

Output projection:
2 * 16 * 64 * 64 = 131,072

Feedforward network, two linear layers:
2 * 16 * 64 * 128 + 2 * 16 * 128 * 64
= 524,288

One encoder layer total:
393,216 + 32,768 + 32,768 + 131,072 + 524,288
= 1,114,112 FLOPs
```

## 3. Approximate full-model totals

```text
MLP:       469,504 FLOPs
CNN:     2,063,488 FLOPs
Attention: 2,329,856 FLOPs
```

## 4. Comparison with THOP

```text
Model       Hand FLOPs    THOP MACs    2*THOP MACs
MLP            469,504      234,752         469,504
CNN          2,063,488    1,031,744       2,063,488
Attention    2,329,856       50,816         101,632
```

For MLP and CNN, `2 * MACs` equals the hand-calculated FLOPs. THOP substantially undercounts Attention because it does not profile the internal operations of PyTorch `TransformerEncoder` by default.
