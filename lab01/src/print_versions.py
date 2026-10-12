import platform
from importlib.metadata import PackageNotFoundError, version

PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "seaborn",
    "torch",
    "torchvision",
    "torchinfo",
    "thop",
    "onnx",
    "onnxruntime",
    "mlflow",
    "memory-profiler",
    "psutil",
    "codecarbon",
    "fastapi",
    "uvicorn",
    "pytest",
    "httpx",
    "locust",
    "requests",
    "pyarrow",
    "joblib",
    "tqdm",
]

print(f"Python=={platform.python_version()}")

for package in PACKAGES:
    try:
        print(f"{package}=={version(package)}")
    except PackageNotFoundError:
        print(f"{package}==NOT INSTALLED")

