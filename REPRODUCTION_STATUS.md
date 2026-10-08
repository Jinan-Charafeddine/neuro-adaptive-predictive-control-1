# Reproduction status

| Component | Status | Required for exact manuscript reproduction |
|---|---|---|
| Clinical pediatric recordings | Not included | Authorized original recordings/features and verified provenance |
| 27-subject clinical results | Not reproduced | Actual participant manifest, labels and predictions |
| Feature dataset | Synthetic demonstration, 27 artificial IDs | Actual processing pipeline and extracted recording features |
| SVR / classical SVM | Implemented, executed | Original tuning grids and selected parameters |
| Statevector-kernel comparison | Implemented, executed; explicitly defined new feature map | Original Qiskit version and feature-map configuration |
| SVR-LSTM | Optional code; syntax checked, not executed | PyTorch, original sequence/target setup, weights and scalers |
| Fuzzy estimator | Implemented, executed on artificial load features | Original independent calibration/evaluation inputs |
| Closed-loop controller | Executable single-joint PD surrogate | Actual OpenSim model and original admittance implementation |
| OpenSim | State export utility only, not executed | Native bindings and original model |
| Tables and figures in this package | Regenerated demo outputs | Original per-run experiment outputs |
| Ethics and consent | No approval is claimed or manufactured | Actual source-study documentation / editor-accepted verification |

Twenty computational realizations quantify numerical variation, not clinical population efficacy. Artificial IDs do not represent CP patients. Keep this distinction in all manuscript data/code statements and repository descriptions.
