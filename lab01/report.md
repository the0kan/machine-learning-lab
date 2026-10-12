# Lab 1: Environment and First System Measurements

## Goal

The goal of this lab was to create a reproducible Python environment, train two baseline classifiers on the Breast Cancer Wisconsin dataset, measure their accuracy and system costs, and evaluate where they could be deployed under the provided Cloud, Edge, Mobile, and TinyML budgets.

## Method

The experiment used Python 3.11.8 and the pinned package versions recorded in `results/versions.txt`. All required random seeds were set to 42. The dataset was loaded with scikit-learn and divided into 70% training and 30% test data using a stratified split with `random_state=42`.

The linear baseline was a `LogisticRegression` model with `max_iter=1000` and `random_state=42`. It was placed after a `StandardScaler` in a pipeline because the unscaled model did not converge within the required 1,000 iterations. The ensemble baseline was a `RandomForestClassifier` with 100 estimators and `random_state=42`.

Training time was measured after one warm-up and reported as the median of five runs. Single-sample inference was warmed up once, repeated 100 times, and reported as the median. Peak process memory was sampled with psutil during separate training and inference runs. Each fitted model was saved with joblib, and its file size was measured in bytes and KB.

The supplied dependency pins contain one conflict: MLflow 2.14.1 declares `pyarrow<16`, while the lab requires PyArrow 16.1.0. The required top-level versions were retained. Neither package is used by the Lab 1 model scripts.

## Results

### Test accuracy

| Model | Test accuracy |
|---|---:|
| Logistic Regression | 0.9883 |
| Random Forest | 0.9357 |

### System measurements

| Model | Median training time (ms) | Median inference time (ms) | Model size (bytes) | Model size (KB) | Training peak memory (MB) | Inference peak memory (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 1.507208 | 0.041896 | 2,286 | 2.232 | 241.656 | 241.688 |
| Random Forest | 58.816459 | 0.778792 | 290,905 | 284.087 | 244.781 | 244.922 |

The memory values are the total resident memory of the Python process, including the interpreter and imported libraries, rather than only the incremental memory allocated by each model.

## Deployment Fit

| Model | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|
| Logistic Regression | Yes | Yes | Yes, with limited memory headroom | No |
| Random Forest | Yes | Yes | Yes, with limited memory headroom | No |

Both models meet the Cloud limits because their measured memory is below 1 GB of available capacity, latency is below 100 ms, and model size is below 500 MB. They also meet the Edge limits: each uses less than the minimum 256 MB device capacity, has latency below 50 ms, and is smaller than 50 MB.

Both models technically meet the supplied Mobile limits. Their inference memory is below 256 MB, latency is below 20 ms, and model size is below 10 MB. However, the measured process memory is close to the 256 MB upper limit, especially for the Random Forest, so a real mobile deployment would require additional profiling with the target runtime and operating-system overhead.

Neither measured Python deployment meets the TinyML memory limit of 256 KB. Logistic Regression satisfies the latency and 100 KB model-size limits, but its measured inference process memory is 241.688 MB. Random Forest also exceeds the 100 KB model-size limit with a size of 284.087 KB and uses 244.922 MB of process memory. Therefore, neither model qualifies for TinyML under the measured configuration.

## Conclusions

1. Logistic Regression produced the highest test accuracy while also training and predicting much faster than Random Forest on this dataset.
2. Model file size and latency alone are not sufficient for deployment decisions; the runtime memory footprint can rule out an otherwise small and fast model.
3. Both models fit the supplied Cloud, Edge, and Mobile budgets in this experiment, but neither measured Python configuration fits the TinyML memory budget.
