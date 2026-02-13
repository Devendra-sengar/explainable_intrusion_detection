# Explainable Hybrid Intelligence for Adaptive Cyber Defense in Distributed Environments

This project implements an adaptive hybrid XGBoost-based Intrusion Detection System (IDS) with explainable AI capabilities for cyber defense in distributed environments.

## Project Structure

```
├── main.py                 # FastAPI application entry point
├── backend/               # Backend API components
│   ├── model_manager.py   # ML model management
│   ├── train_initial_model.py  # Initial model training
│   └── requirements.txt   # Backend-specific dependencies
├── frontend/              # Frontend components
├── data/                  # Data files and models
├── archive (30)/          # CICIDS2017 dataset files
└── requirements.txt       # Main project dependencies
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "Explainable Hybrid Intelligence for Adaptive Cyber Defense in Distributed Environments"
```

### 2. Create Virtual Environment

#### On Windows (PowerShell):
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Verify activation (you should see (venv) prefix)
```

#### On Windows (Command Prompt):
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate.bat
```

#### On macOS/Linux:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install all project dependencies
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Check installed packages
pip list

# Test FastAPI installation
python -c "import fastapi; print('FastAPI installed successfully')"

# Test ML libraries
python -c "import scikit_learn, xgboost, pandas; print('ML libraries installed successfully')"
```

## Running the Application

### Start the API Server

```bash
# Make sure virtual environment is activated
# Run the FastAPI server
python main.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive API Documentation**: http://localhost:8000/docs
- **Alternative API Documentation**: http://localhost:8000/redoc

### Using Jupyter Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# Or start Jupyter Notebook
jupyter notebook
```

## Deactivating Virtual Environment

When you're done working on the project:

```bash
# Deactivate the virtual environment
deactivate
```

## Project Features

- **Adaptive Learning**: Real-time model updates based on new threat data
- **Hybrid Intelligence**: Combines multiple ML techniques for enhanced detection
- **Explainable AI**: Provides interpretable results for security analysts
- **RESTful API**: Easy integration with existing security infrastructure
- **Web Interface**: User-friendly dashboard for model interaction

## API Endpoints

- `POST /predict` - Get predictions for network traffic samples
- `POST /adaptive_update` - Update model with new labeled data
- `GET /model_info` - Get current model statistics and performance metrics
- `GET /health` - Health check endpoint

## Contributing

1. Ensure virtual environment is activated
2. Install development dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest` (if tests are available)
4. Follow code formatting: `black .` (if black is installed)

## Troubleshooting

### Virtual Environment Issues

- **Permission errors on Windows**: Run PowerShell as Administrator
- **Script execution disabled**: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- **Python not found**: Ensure Python is added to your PATH environment variable

### Dependency Issues

- **Package installation fails**: Try `pip install --upgrade pip` first
- **Version conflicts**: Use `pip install --force-reinstall <package-name>`
- **Memory issues**: Install packages individually for large dependencies

### Model Training Issues

- **Data not found**: Ensure CICIDS2017 dataset is in the `archive (30)/` folder
- **Memory errors**: Reduce batch sizes or sample smaller datasets
- **CUDA issues**: Install CPU-only versions if GPU is not available

## System Requirements

- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: At least 5GB free space for datasets and models
- **CPU**: Multi-core processor recommended for model training