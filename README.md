## Forecasting Software Runtime Metrics: A Comparative Study of Classical Statistical, Neural Network, and Foundation Models

### Requirements

The experiment can be executed entirely on a CPU. However, utilizing a GPU with CUDA can significantly speed up the process.

- The experiment has been conducted using Python 3.8. The repository includes a Dockerfile that builds an image based on *python:3.8-bookworm* and installs all required dependencies.
- The required Python packages are listed in *requirements.txt* file. 

---
### Experimental Setup
 
- Install the recommended version of **Python 3**.
- Clone the repository
- Navigate the repository root: `cd repository-folder`
- Create a virtual environment and activate it: `python3 -m venv .venv`, `source .venv/bin/activate`
- Install the required dependencies: `pip install -r requirements.txt`

---
### Usage

#### Experiment pipeline

The `cfg.py` script contains the constants configuration for setting up the experimental procedure, such as input/output folders and window size.

The experimental pipeline can be executed through the `exec.sh` script. The script will execute the following steps:
- Parameters estimation for classical statistical models
- Baseline experiments with sNaive and sMM
- Classical statistical models experiments
- Recurrent Neural Networks experiments
- Foundation Models experiments

---

### Licensing and Third-Party Components

This replication package is distributed under MIT License.

It mainly uses the following third-party components:

- **PyTorch** — https://pytorch.org/
- **TensorFlow** — https://www.tensorflow.org/
- **Chronos Forecasting** https://github.com/amazon-science/chronos-forecasting
- **TimesFM** — https://github.com/google-research/timesfm/
