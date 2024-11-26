# CNN in CUDA

We follow the same commands as given on Piazza.

## Subtask 1: Convolution and Activation Functions

To run subtask 1, execute the following command:

```bash
./subtask1 [task of choice]
```

- **Task Choices**:
  1. Convolution
  2. Non-linear Activations
  3. Subsampling
  4. Converting a Vector

### Convolution Example

```bash
./subtask1 1 [N] [M] [P] [matrix values...]
```

- `[N]` - Size of the square matrix.
- `[M]` - Size of the kernel.
- `[P]` - Padding value.
- Matrix and kernel values are space-separated.

Example:

```bash
./subtask1 1 15 5 0 [matrix values...]
```

### Activation Function Example

Specify the activation function (0=ReLU, 1=Tanh), followed by matrix size and values:

```bash
./subtask1 2 [activation function] [N] [M] [matrix values...]
```

### Subsampling Example

Choose the pooling function (0=Max pool, 1=Avg pool), specify the pooling size, matrix size, and values:

```bash
./subtask1 3 [pooling function] [pooling size] [N] [matrix values...]
```

### Vector Conversion Example

Specify the function (0=Sigmoid, 1=Softmax), followed by vector values:

```bash
./subtask1 4 [function] [vector values...]
```
Example
```bash
./subtask1 4 1 "1.0 2.0 3.0"
```

## Subtask 2: Advanced Convolution

Follows the same execution format as subtask 1:

```bash
./subtask2 [task of choice]
```

## Subtask 3: Image Processing

This task assumes files have been preprocessed using `preprocess.py`. Run the task without additional parameters:

```bash
./subtask3
```

The code processes the first file from `pre-proc-img` named `000000-num7.txt`, runs the file, and prints the top 5 probabilities in the `output` folder, maintaining the input file's name.

## Subtask 4: Stream Processing

To execute subtask 4, specify if you want to run with streams:

```bash
./subtask4 [1 - with streams, 0 - without streams]
```

## Directory Structure

```
kerbno1_kerbno2/
├── README.md
├── img/                  # Images directory
├── weights/              # Weights directory
├── pre-proc-img/         # Preprocessed data directory
├── preprocessing.py  # Pre-processing source code
├── report.pdf            # Project report
├── output/               # Output directory
└── src/                  # Source code directory
    ├── assignment2_subtask1.cpp
    ├── assignment2_subtask2.cu
    ├── assignment2_subtask3.cu
    └── assignment2_subtask4.cu
```