# Neuro-Adaptive Control: Executable Demonstration and Data Templates

This repository contains a new, runnable implementation of selected learning and signal-processing modules described in the manuscript, together with a reduced-order controller demonstration. **It does not reproduce the manuscript's clinical dataset, OpenSim closed-loop experiment, or numerical results.** The original recordings, original controller code, model files and exact experimental configuration were not supplied.

## Included data

- `data/demo_features.csv`: 4,779 directly generated feature windows from 27 artificial IDs, split 19 training / 4 validation / 4 test IDs. No children, diagnoses, consent or hospital recordings are represented by these IDs.
- `data/demo_subjects.csv`: artificial-ID split manifest.
- `data/Dataset_Demo_and_Template.xlsx`: complete demo features, split manifest, data dictionary and empty real-feature import template. The demo does not provide raw 1000-Hz sEMG; rows are precomputed artificial features at 100-ms intervals.
- `data/real_features_template.csv` and `data/real_subjects_template.csv`: header-only templates. Missing real data remain empty.
- `results/`: outputs calculated by the supplied scripts using demo data. Values are not copied from the article and must not be reported as pediatric CP validation.

## Quick start

Python 3.11 or newer is recommended. From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python src/experiment.py
python -m unittest discover -s tests -v
```

The default command regenerates the artificial dataset with seed 2026, fits models and writes results and figures. Do not place private recordings in the demo files. Use a separate private path and output folder for authorized data:

```bash
python src/experiment.py --data /private/path/authorized_features.csv --output results/private_run
```

The controller and fuzzy demonstrations remain synthetic even when real offline features are supplied. The Excel file is a snapshot of the included data; if the generator or data changes, it must be rebuilt rather than treated as live.

## Implemented and tested core

- RBF SVR with a scaling pipeline and five-fold participant-grouped hyperparameter search.
- Classical SVM and a four-qubit statevector fidelity-kernel SVM, with matched balanced training rows and validation-only C selection.
- A NumPy feature map explicitly defined in `src/experiment.py`: two repeats of Hadamard gates and diagonal Z/nearest-neighbor ZZ phases. This is not asserted to be identical to the article's Qiskit ZZFeatureMap settings. No quantum hardware or quantum advantage is claimed.
- Training-only fitted standardization and angular scaling. The test set is not used for tuning or balancing.
- Confusion matrices, per-class precision/recall/F1, aggregate accuracy and subject-cluster bootstrap intervals. These intervals are descriptive and particularly fragile with four test IDs.
- Mamdani fuzzy load estimation with the manuscript's stated membership shapes and rule table. Demo load inputs are artificially related to known load; this is not clinical or independent estimator validation.
- Twenty paired-seed runs of an explicit single-joint PD controller surrogate, with noise and burst/fatigue-like perturbations, pre-clipping joint-limit counts and torque-saturation counts. This is **not** OpenSim, a Hill-muscle model, or the manuscript's admittance law. Its muscle proxies are constructed from residual torque demand, not physiological measurements.

## Optional SVR-LSTM

```bash
python -m pip install -r requirements-lstm.txt
python src/lstm_optional.py --data data/demo_features.csv --epochs 100
```

This provides a two-layer LSTM with 64 hidden units, dropout 0.2, Adam 0.001, batch size 64 and validation early stopping (patience 10). Input sequences contain five consecutive windows from the same subject and trial. The elbow-only hybrid uses grouped cross-fitted SVR estimates and a 100-ms future-angle target. It is a new example architecture, not the original trained model. PyTorch is unavailable in the build environment, so this optional training path was syntax-checked but not executed. It saves weights and test predictions; deploying the weights also requires preserving scalers and the feature-processing configuration.

## OpenSim export utility

```bash
python src/opensim_export.py /path/to/original_model.osim
```

Install the OpenSim 4.3 native Python bindings according to the official OpenSim instructions. The utility exports coordinates and muscle activations from an author-supplied model; it does not reconstruct the unpublished closed-loop controller. OpenSim and the original `.osim` file were unavailable during testing, so the utility was syntax-checked only.

## Using real recordings

1. Confirm institutional permission, original ethics/consent status and authorization for secondary analysis before using or sharing recordings.
2. Keep clinical recordings outside this public repository. Replace neither a missing participant nor a missing diagnostic field with invented data.
3. Extract four causal sEMG feature channels and synchronized joint-angle targets using the actual acquisition protocol. Populate pseudonymous `subject_id`, `trial_id`, `window_index`, `time_s`, intention and future-angle fields. Use the label `clinical_recording_features` only for actual recording-derived features.
4. Assign every subject to exactly one split. Do not create sequences across trial boundaries, split overlapping windows across subsets, or fit normalization on held-out subjects.
5. The CSV contains model inputs, not provenance or ethics documentation. Keep a separate controlled-access dataset record describing acquisition, de-identification, diagnoses, sampling and approvals.
6. Inspect regenerated metrics and saved predictions before making research claims. The existing demo metrics cannot be reused as clinical results.

## Exact article reproduction requires

- Original authorized recordings or shareable feature matrices and participant/trial manifest.
- Original preprocessing implementation, labels, feature definitions and normalization parameters.
- Exact training/tuning configuration, seeds, trained weights and original Qiskit feature-map settings.
- Original OpenSim model, excitation/control inputs, task trajectories, controller law, muscle properties and limits.
- Original noise/fatigue/burst/load schedules and all per-run outputs supporting tables and figures.

See `docs/REPRODUCTION_STATUS.md` for scope. There is no claim that this package resolves the editor's missing ethics or reproducibility information.

## Upload to GitHub

Create a repository named `NeuroAdaptive-Control-Demo`. Extract this package, open **Add file → Upload files**, and upload its contents including `src`, `data`, `docs`, `tests`, `results` and the README. Check that no confidential files have been added before making it public. The package is prepared locally; it has not been published to a GitHub account.

## License and dependencies

No license granting rights to clinical data is included. Confirm ownership and choose an appropriate code license before public release. Core dependency versions tested during the build are recorded in `requirements-tested.txt` and `results/metrics.json`. Numerical details can vary across library/platform versions.

### Optional raw-recording importer

`src/preprocess_raw.py` accepts a privately supplied, uniformly sampled 1000-Hz CSV with synchronized angle/label columns. It applies causal 20–450-Hz bandpass filtering, rectification, 5-Hz lowpass filtering, 200-ms windows and 100-ms stride. Normalization uses outer-training-only peaks. This is an example preprocessing choice; strict fold-specific normalization requires redoing peak fitting inside each CV fold. Filter warm-up and original synchronization must be specified before research use. This importer is syntax-checked only because no real raw recordings were supplied.

```bash
python src/preprocess_raw.py /private/path/raw_aligned.csv --output data/private/features.csv
```
