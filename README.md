# MedHE: Communication-Efficient Privacy-Preserving Federated Learning with Adaptive Gradient Sparsification

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-red.svg)
![TenSEAL](https://img.shields.io/badge/TenSEAL-0.3.16-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**MedHE** is a novel federated learning framework that integrates adaptive gradient sparsification with CKKS homomorphic encryption to enable privacy-preserving collaborative learning on sensitive healthcare data. This implementation demonstrates state-of-the-art results in healthcare NLP while maintaining strong privacy guarantees.

## 📖 Paper Abstract
Healthcare federated learning requires strong privacy guarantees while maintaining computational efficiency across resource-constrained medical institutions. MedHE introduces a dynamic threshold mechanism for top-k gradient selection, achieving **90% communication reduction** while preserving model utility. Our approach provides formal security proofs under RLWE assumptions and demonstrates **differential privacy guarantees with ε ≤ 1.5**. Comprehensive evaluation shows strong privacy preservation against five attack vectors with **membership inference attack success rates near random guessing (50.1%)**.

## 🚀 Key Features

### 🔒 Privacy & Security
- **Homomorphic Encryption**: CKKS implementation with semantic security guarantees
- **Adaptive Gradient Sparsification**: Dynamic top-k selection with 90% compression
- **Formal Security Proofs**: RLWE-based security analysis and differential privacy
- **Attack Resistance**: Comprehensive evaluation against 5 attack vectors

### ⚡ Performance
- **91.1% Accuracy** on healthcare text classification
- **90% Communication Reduction** vs standard federated learning
- **54% Computational Overhead** with linear scalability to 100+ clients
- **HIPAA Compliance** ready for real-world healthcare deployment

### 🏥 Healthcare Focus
- **Drug Review Analysis**: UCI Drug Review dataset (4,142 samples)
- **DistilBERT Integration**: Efficient transformer model for medical NLP
- **Non-IID Robustness**: Realistic medical institution data distribution

## 📊 Results Summary

| Metric | Result |
|--------|---------|
| Accuracy | 91.1% |
| F1 Score | 0.950 |
| Communication Reduction | 90% |
| MIA Success Rate | 50.1% (≈ random) |
| Differential Privacy ε | 1.500 |
| Training Overhead | +54% |

## 🛠️ Quick Start

### Installation

git clone https://github.com/Farjana-Yesmin/MedHE.git
cd MedHE

# Install dependencies
pip install transformers==4.41.2 torch==2.3.0 pandas==2.2.2 \
ucimlrepo tenseal==0.3.16 scikit-learn==1.5.0 matplotlib seaborn

Running the Framework
# Open and execute the main notebook
jupyter notebook satml-2026.ipynb

# Or run specific analyses directly
python run_analysis.py --analysis baseline_comparison
python run_analysis.py --analysis privacy_evaluation

Basic Usage
from medhe_framework import MedHEFL

# Initialize MedHE framework
fl_framework = MedHEFL(
    model_name='distilbert-base-uncased',
    num_clients=5,
    sparsity_level=0.9,
    encryption_enabled=True
)

# Run federated learning
results = fl_framework.train(
    num_rounds=3,
    epochs_per_client=2,
    learning_rate=1e-4
)

# Evaluate privacy and performance
privacy_metrics = fl_framework.evaluate_privacy()
performance_metrics = fl_framework.evaluate_performance()

MedHE/
├── satml-2026.ipynb              # Main implementation notebook
├── medhe_framework/              # Framework source code
│   ├── __init__.py
│   ├── federated_learning.py     # Core FL implementation
│   ├── encryption.py            # CKKS homomorphic encryption
│   ├── sparsification.py        # Adaptive gradient sparsification
│   └── security_analysis.py     # Privacy attack evaluation
├── papers/
│   ├── medhe_satml2026.pdf      # Conference paper
│   └── medhe_technical.pdf      # Technical report
├── results/                     # Experimental results
│   ├── communication_analysis.png
│   ├── privacy_evaluation.png
│   └── scalability_analysis.png
├── requirements.txt             # Python dependencies
└── README.md

🎯 Reproducibility


Environment Setup


Python: 3.11.13
PyTorch: 2.3.0
Transformers: 4.41.2
TenSEAL: 0.3.16 (Homomorphic Encryption)
Hardware: Tesla V100 GPU (16GB) recommended

Dataset

UCI Drug Review Dataset (ID 461)
4,142 patient reviews for drug effectiveness classification
Binary classification: effective vs. ineffective treatments
Hyperparameters
{
    "learning_rate": 1e-4,
    "batch_size": 8,
    "sparsity_level": 0.9,
    "num_clients": 5,
    "fl_rounds": 3,
    "epochs_per_client": 2,
    "non_iid_alpha": 0.1
}


🔬 Experimental Evaluation

Privacy Attacks Evaluated

Membership Inference Attacks (MIA)
Model Inversion Attacks
Property Inference Attacks
Gradient Leakage Attacks
Eavesdropping Attacks
Performance Metrics

Accuracy & F1 Score: Model utility preservation
Communication Efficiency: Bytes transmitted per round
Computational Overhead: Training time comparison
Scalability Analysis: Performance with increasing clients

🏛️ Regulatory Compliance

MedHE is designed to support HIPAA compliance with:

End-to-end encryption of all communications
Formal differential privacy guarantees (ε ≤ 1.5)
Comprehensive audit trails and access controls
Secure key management protocols
📈 Applications

Healthcare Use Cases

Multi-Hospital Collaborative Learning: COVID-19 outcome prediction across 10+ hospitals
Pharmaceutical Research: Secure drug discovery without revealing proprietary compounds
Public Health Surveillance: Population-level insights with individual privacy preservation

Extended Applications

Financial Services: Secure fraud detection across banks
Legal AI: Confidential document analysis
Government Analytics: Privacy-preserving public data analysis

🔮 Future Work

Post-Quantum Security: Integration with lattice-based cryptography
Malicious Adversary Protection: Byzantine-robust aggregation
Dynamic Sparsity Adjustment: Adaptive sparsity based on convergence metrics
Hardware Acceleration: FPGA/GPU optimization for CKKS operations

📚 Citation

If you use MedHE in your research, please cite our SATML 2026 paper:
@inproceedings{medhe2026,
  title={MedHE: Communication-Efficient Privacy-Preserving Federated Learning with Adaptive Gradient Sparsification},
  author={Farjana Yesmin},
  booktitle={IEEE Conference on Secure and Trustworthy Machine Learning (SATML)},
  year={2026},
  pages={1--10}
}

🤝 Contributing

We welcome contributions! Please see our Contributing Guidelines for details.

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

🆘 Support

For questions and support:

📧 Email: Farjanayesmin76@gmail.com
💬 Issues: GitHub Issues
📖 Documentation: Full Documentation

MedHE: Building trustworthy AI for healthcare through privacy-preserving federated learning. 🛡️⚕️🤖


