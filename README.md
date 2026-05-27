FEDERATED LEARNING WITH DIFFERENTIAL PRIVACY
=============================================

This project implements a federated learning system with differential privacy on 
healthcare data (MIMIC-IV). The system trains a neural network model on distributed 
data while preserving privacy through differential privacy mechanisms.


PROJECT OVERVIEW
================

The project implements a client-server architecture where:

* A central server coordinates the federated learning process
* Multiple clients (10 in this implementation) train the model on their local data
* Differential privacy is applied to protect sensitive information during training
* Model accuracy maintained at 85.38% across all privacy budgets (epsilon 0.31 to infinity)


DATASET
=======

The project uses a healthcare dataset (MIMIC-IV) containing the following features:

* ABP Diastolic
* ABP Systolic
* Glucose
* Heart Rate
* Respiratory Rate
* Temperature (°F)
* hospital_expire_flag (target variable - mortality prediction)

The dataset contains 3,000 patient records.


RESULTS
=======

PRIVACY-UTILITY TRADEOFF
------------------------

Privacy Budget (Epsilon) | Noise Level | Test Accuracy
inf (non-private)        | 0.0         | 85.38%
7.85                     | 0.5         | 85.38%
0.92                     | 1.0         | 85.38%
0.46                     | 1.5         | 85.38%
0.31 (strong privacy)    | 2.0         | 85.38%

Key finding: Differential privacy with epsilon < 1.0 (strong formal privacy) 
achieves zero accuracy loss.


FEDERATED TRAINING CONVERGENCE
-------------------------------

Round 1: 0.8552 accuracy
Round 2: 0.8553 accuracy
Round 3: 0.8562 accuracy
Round 4: 0.8560 accuracy
Round 5: 0.8577 accuracy


PROJECT STRUCTURE
=================

.
├── preprocessing.py
├── federated_client.py
├── differential_privacy_client.py
├── server.py
├── federated_pipeline.py
├── benchmarking_suite.py
├── technical_report.md
├── requirements.txt
├── .gitignore
└── README.md


REQUIREMENTS
============

* Python 3.9+
* PyTorch 2.0+
* Opacus 1.4.0+
* scikit-learn
* pandas
* numpy
* matplotlib


INSTALLATION
============

1. Clone the repository:

git clone https://github.com/shrawani-gulhane/federated_learning_privacy.git
cd federated_learning_privacy

2. Create virtual environment:

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

3. Install required packages:

pip install -r requirements.txt


USAGE
=====

RUN FEDERATED LEARNING PIPELINE
--------------------------------

python federated_pipeline.py

Trains 10 heterogeneous clients for 5 federated rounds with server-side aggregation.


RUN PRIVACY-UTILITY BENCHMARK
------------------------------

python benchmarking_suite.py

Tests accuracy across 5 privacy levels. Generates:
* privacy_utility_curve.png - Privacy-utility tradeoff visualization
* benchmark_results.csv - Detailed results


IMPLEMENTATION DETAILS
======================

FEDERATED LEARNING
- Algorithm: FedAvg (Federated Averaging)
- Clients: 10 heterogeneous healthcare institutions
- Server: Aggregates model updates via weight averaging
- Local training: 5 epochs per round

DIFFERENTIAL PRIVACY
- Framework: Opacus (PyTorch)
- Mechanism: Per-sample gradient clipping + Gaussian noise
- Max gradient norm: 1.0
- Privacy accounting: Renyi Differential Privacy

MODEL
- Architecture: 2-layer MLP (7 -> 64 -> 32 -> 1)
- Task: Binary mortality prediction
- Optimizer: SGD (lr=0.01)
- Loss: Binary cross-entropy


CONTRIBUTING
============

1. Fork the repository
2. Create your feature branch (git checkout -b feature/improvement)
3. Commit your changes (git commit -m 'Add improvement')
4. Push to the branch (git push origin feature/improvement)
5. Open a Pull Request


LICENSE
=======

MIT License - see LICENSE file for details


ACKNOWLEDGMENTS
===============

* Opacus library for PyTorch differential privacy
* MIMIC-IV dataset from PhysioNet
* McMahan et al. (2017) for FedAvg algorithm
* Abadi et al. (2016) for differential privacy foundations
