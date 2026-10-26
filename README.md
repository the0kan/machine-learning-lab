# Machine Learning Systems Laboratory

Coursework repository for a machine learning systems laboratory. The work focuses not only on model accuracy, but also on reproducibility, benchmarking, resource usage, and deployment constraints.

The repository will contain 11 laboratory assignments and a final project. Each lab keeps its implementation, generated evidence, and report together.

## Progress

| Work | Status | Focus |
|---|---|---|
| [Lab 01](lab01/) | Complete | Reproducible environment, baseline models, system measurements, and deployment fit |
| [Lab 02](lab02/) | Complete | NumPy neural network, gradient checking, hyperparameter comparison, and timing |
| Labs 03-11 | Planned | Added as the corresponding lab instructions are released |
| [Final project](project/) | Planned | Dataset, model, data card, model card, and system results |

## Lab 01 at a glance

Lab 01 uses the scikit-learn Breast Cancer Wisconsin dataset and a reproducible 70/30 stratified split (`random_state=42`). It compares:

- Logistic Regression with feature standardization
- Random Forest with 100 estimators

### Results

Measurements were collected on the local development machine. Training time is the median of five runs after one warm-up. Single-sample inference time is the median of 100 runs after one warm-up.

| Model | Test accuracy | Median training | Median inference | Saved model size |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9883 | 1.507 ms | 0.042 ms | 2.232 KB |
| Random Forest | 0.9357 | 58.816 ms | 0.779 ms | 284.087 KB |

Under the assignment's stated limits, both measured configurations fit the Cloud, Edge, and Mobile budgets. Neither fits the TinyML memory budget. See the [complete Lab 01 report](lab01/report.md) for the methodology, memory measurements, limitations, and deployment reasoning.

## Repository structure

```text
machine-learning-lab/
├── README.md
├── requirements.txt
├── data/                 # Local datasets; excluded from Git
├── lab01/
│   ├── report.md
│   ├── src/
│   └── results/
├── lab02/ ... lab11/
│   ├── report.md
│   ├── src/
│   └── results/
└── project/
    ├── README.md
    ├── src/
    ├── results/
    ├── data_card.md
    └── model_card.md
```

## Reproduce Lab 01

### Requirements

- Python 3.11.8
- Git
- macOS, Linux, or Windows

Create and activate an isolated environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with:

```powershell
.venv\Scripts\activate
```

Install the exact top-level versions required by the lab manual:

```bash
python -m pip install --upgrade pip
python -m pip install --use-deprecated=legacy-resolver -r requirements.txt
python -m pip install packaging==24.2 protobuf==4.25.9 pytz==2024.2
```

Run the Lab 01 workflow:

```bash
python lab01/src/print_versions.py > lab01/results/versions.txt
python lab01/src/train_baselines.py
python lab01/src/benchmark_models.py
```

Generated evidence is stored in [`lab01/results/`](lab01/results/):

- `versions.txt` - Python and package versions
- `baseline_accuracy.csv` - test accuracy for both models
- `system_measurements.csv` - timing, model size, and peak process memory
- `logistic_regression.joblib` and `random_forest.joblib` - fitted models

## Reproduce Lab 02

Open [`lab02/Machine_Learning_Lab2.ipynb`](lab02/Machine_Learning_Lab2.ipynb) with the repository's Python 3.11.8 environment and run all cells from top to bottom. The notebook performs the dataset split, forward and backward propagation checks, numerical gradient check, six training runs, accuracy comparison, and timing measurements.

Refresh the environment evidence with:

```bash
python lab02/src/print_versions.py > lab02/results/versions.txt
```

Generated evidence is stored in [`lab02/results/`](lab02/results/):

- `versions.txt` - Python and package versions
- `loss_curves.png` - per-epoch cross-entropy loss for all six configurations
- `timing_breakdown.txt` - median forward and backward times
- `timing_breakdown.png` - visual comparison of the timing measurements

See the [complete Lab 02 report](lab02/report.md) for the methodology, accuracy table, gradient-check evidence, timing interpretation, and conclusions.

## Reproducibility rules

- Random seeds are fixed at 42 for Python, NumPy, and PyTorch.
- The train/test split is stratified and fixed with `random_state=42`.
- Timing results use a warm-up followed by repeated measurements.
- Reports contain results and explanations; executable code stays under `src/`.
- Every completed lab is committed separately so that the Git history documents the work.

## Known dependency conflict

The required package list pins both `mlflow==2.14.1` and `pyarrow==16.1.0`, while MLflow 2.14.1 declares `pyarrow<16`. The repository retains the exact top-level versions specified by the lab manual. Lab 01 does not import MLflow or PyArrow, so this conflict does not affect its experiments. It is also documented in the Lab 01 report.

## Documentation

- [Lab 01 report](lab01/report.md)
- [Lab 02 report](lab02/report.md)
- [Pinned dependencies](requirements.txt)

## Author

Okan Ozkan
