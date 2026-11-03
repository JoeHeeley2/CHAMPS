# CH Arylation: Machine Prediction of Selectivity (CHAMPS)

**Paper:** [insert paper ref here]

CHAMPS is a computational workflow and set of machine learning models for predicting the
site-selectivity of C–H arylation on heteroaromatic ring systems. Each candidate site on a
heterocycle is described using three computed reactivity properties:

| Property | What it describes | How it is computed | ML target flag |
|---|---|---|---|
| **Acidity** | Ease of C–H deprotonation at each C–H site | Gaussian 16 DFT: energy of each deprotonated anion vs. the neutral heterocycle | `HCS` (value stored on the H atom) |
| **Nucleophilicity** | Electrophile (Br⁺) affinity at each C–H carbon | Gaussian 16 DFT: energy of each brominated intermediate vs. the neutral heterocycle | `CCS` (value stored on the C atom) |
| **Donicity** | Ability of ring nitrogens to coordinate/direct Pd | ORCA DFT: Pd exchange energies for N-palladated structures | `NCS` (value stored on the N atom) |

Acidity and nucleophilicity can be expressed relative to a reference heterocycle,
pyrazolo[1,5-a]pyrimidine. Together they place each site on an "activation map", which shows
which sites are likely to undergo C–H arylation.

Running DFT on every heterocycle is expensive, so CHAMPS trains machine learning models to
predict these properties directly from 3D structure. They are Kernel Ridge Regression models
on DScribe descriptors (local many-body tensor representation, LMBTR), plus a classifier that
predicts whether a site is reactive. The models work on structures optimised with DFT or with
cheap molecular mechanics (MM).

The heterocycles come from the **VEHICLe** database (Virtual Exploratory Heterocyclic Library),
which is included in `data/`.

---

## Contents

1. [Repository layout](#repository-layout)
2. [Installation and dependencies](#installation-and-dependencies)
3. [About the included `mol_translator` and IMPRESSION code](#about-the-included-mol_translator-and-impression-code)
4. [Use the workflow wrappers](#use-the-workflow-wrappers)
5. [Data analysis: acidity and nucleophilicity](#data-analysis-acidity-and-nucleophilicity)
6. [Data analysis: donicity](#data-analysis-donicity)
7. [Machine learning](#machine-learning)
8. [Plots](#plots)
9. [Known issues and caveats](#known-issues-and-caveats)
10. [Citation](#citation)

---

## Repository layout

```
CHAMPS/
├── data/                          # Input data
│   ├── VEHICLe.csv                # Full VEHICLe database (Regid, SMILES, hit counts, Pgood, ...)
│   ├── dj1.csv                    # 'dj1' subset of VEHICLe used in the paper
│   ├── dj1_atom_df.csv            # ML atom dataframe for dj1 (types, xyz, connectivity, property in 'shift')
│   └── dj1_pair_df.csv            # ML pair dataframe for dj1 (atom-pair distances, path lengths)
│
├── data_analysis/                 # Structure generation, DFT input/output handling, property calculation
│   ├── workflow.py                # >>> HIGH-LEVEL WORKFLOW WRAPPERS: start here <<<
│   ├── VEHICLe_read.py            # Read VEHICLe SMILES into RDKit mol objects
│   ├── VEHICLe_filters.py         # Filters used to build dataset subsets (dj1, dj2, djr, donicity, ...)
│   ├── chembl_filters.py          # Filters applied to ChEMBL data
│   ├── neutral_generation.py      # Gaussian inputs for neutral ("Base_het") heterocycles
│   ├── anion_generation.py        # Gaussian inputs for each C–H deprotonated anion (acidity)
│   ├── bromine_generation.py      # Gaussian inputs for each brominated intermediate (nucleophilicity)
│   ├── add_substituent.py         # Add substituents to heterocycles and (optionally) write inputs
│   ├── nitrogen_palladation.py    # Build N-palladated structures for donicity calculations
│   ├── file_generation.py         # xyz/ORCA input writers (incl. donicity opt + freq inputs)
│   ├── logfile_read.py            # Read SCF / thermochemistry energies from Gaussian and ORCA output
│   ├── calculation_check.py       # Flag failed calculations and structures that drifted from the input
│   ├── property_calculate.py      # Compute acidity, nucleophilicity, relative values, ΔG
│   ├── donicity_data_extraction.py# ORCA output checks, corrected free energies, ΔG_exch (donicity)
│   ├── impression_input.py        # Build ML atom/pair dataframes and classifier inputs
│   ├── sampling.py                # Tanimoto-similarity diversity sampling, random/stratified sampling
│   ├── mol_mechanics.py           # MM (MMFF) structure generation for ML input
│   ├── regression.py              # Regression of experimental yields against computed properties
│   ├── yield_v_properties.py      # Donicity ΔG vs ΔE comparison / yield analysis script
│   ├── plots.py                   # Activation maps and other data analysis plots
│   ├── pd_plots.py                # 3D plots that include Pd exchange energies (donicity)
│   ├── print_file.py              # Helpers to write dataframes to .csv and SMILES to grid images
│   └── hyperparameter_tuning/     # Hyperparameter search routines (from IMPRESSION)
│       ├── HPS_gaussian.py        # Bayesian (Gaussian process) search; used by the ML models
│       ├── HPS_grid.py            # Grid search
│       └── HPS_random.py          # Random search
│
├── machine_learning/              # Model training, prediction and evaluation
│   ├── model/
│   │   ├── dscribe_model.py       # DSCRIBEmodel: KRR on DScribe descriptors (acidity/nucleophilicity/donicity)
│   │   ├── classifier_model.py    # ClassifierModel: site reactivity classifiers (RF, SVC, MLP, ...)
│   │   ├── fchl_model.py          # FCHLmodel: FCHL/KRR model (from IMPRESSION)
│   │   ├── GDCNN.py               # Graph convolutional network architecture (PyTorch Geometric)
│   │   └── features/
│   │       ├── dscribe_inputs.py  # SOAP, ACSF, Coulomb matrix, (L)MBTR descriptor generation
│   │       └── fchl_input.py      # FCHL representation generation (from IMPRESSION)
│   ├── trained_models/            # Trained CHAMPS models (pickled), see below
│   ├── imp_train.py               # Train-and-predict / load-and-predict for FCHL models
│   ├── hyperparam_opt.py          # Cross-validated hyperparameter optimisation
│   ├── total_prediction.py        # Full CHAMPS prediction: KRR property predictions -> classifier
│   ├── molecular_mechanics.py     # Full CHAMPS prediction starting from SMILES via MM structures
│   ├── top_n_accuracy.py          # Top-n ordering accuracy of predicted reactive sites
│   └── plots.py                   # DFT vs. predicted and error-distribution plots
│
├── mol_translator/                # Bundled (old) copy of the Butts group mol_translator package
├── requirements.txt
└── README.md
```

---

## Installation and dependencies

### Python environment

The code was developed against the package versions in `requirements.txt`. It uses
`DataFrame.append`, which was removed in pandas 2.0, so you **must** use pandas < 2.0.
A Python 3.8–3.10 environment is recommended.

```bash
conda create -n champs python=3.9
conda activate champs

# Chemistry toolkits (most easily installed with conda)
conda install -c conda-forge rdkit openbabel

# Pinned Python packages
pip install -r requirements.txt
```

### Dependencies

`requirements.txt` lists:

- `matplotlib`, `seaborn`: plotting
- `pandas` (~1.5), `numpy`: data handling
- `scikit-learn`: KRR, classifiers, metrics, data splitting
- `tqdm`: progress bars
- `qml`: FCHL representations and kernels (IMPRESSION FCHL model)
- `dscribe`, `ase`: SOAP / ACSF / Coulomb matrix / MBTR / LMBTR descriptors
- `networkx`: molecular graphs

These are imported but **not** in `requirements.txt`, so install them separately:

- **`mol_translator`**: Butts group molecule conversion package. An old copy is bundled in this
  repo; see [the next section](#about-the-included-mol_translator-and-impression-code).
- `rdkit`: SMILES handling, 3D embedding, MMFF optimisation, fingerprints
- `openbabel` / `pybel`: used inside `mol_translator`
- `scikit-optimize` (`skopt`): Bayesian hyperparameter search (`BayesSearchCV`)
- `bayesian-optimization` (`bayes_opt`): Gaussian process hyperparameter search in
  `hyperparameter_tuning/HPS_gaussian.py`
- `torch` and `torch_geometric`: only needed for the graph models (`GDCNN.py`, graph features in
  `classifier_model.py`)

### External quantum chemistry software

You only need these to generate *new* DFT data. The trained ML models do not need them:

- **Gaussian 16**: acidity and nucleophilicity calculations
- **ORCA**: donicity (Pd exchange) calculations

### Making `mol_translator` importable

The scripts import `mol_translator` as a top-level package (`from mol_translator import aemol`).
The bundled copy sits at `mol_translator/mol_translator/`, so add the outer `mol_translator/`
directory to your `PYTHONPATH`:

```bash
# Linux / macOS
export PYTHONPATH="$PYTHONPATH:/path/to/CHAMPS/mol_translator"
```

```powershell
# Windows PowerShell
$env:PYTHONPATH = "$env:PYTHONPATH;C:\path\to\CHAMPS\mol_translator"
```

Scripts in `data_analysis/` and `machine_learning/` import their sibling modules directly
(for example `from workflow import ...` and `from model.dscribe_model import DSCRIBEmodel`). Run
them from inside their own directory, or add that directory to `PYTHONPATH` too.

The `hyperparameter_tuning` package is in `data_analysis/`, but `machine_learning/` code imports
it (`from hyperparameter_tuning import HPS_gaussian`). To run the ML code, add `data_analysis/`
to your `PYTHONPATH`.

---

## About the included `mol_translator` and IMPRESSION code

> **The `mol_translator` package and the IMPRESSION machine learning code used in this project
> are not my own work. They are taken from the Butts group GitHub:**
>
> - mol_translator: https://github.com/Buttsgroup/mol_translator
> - IMPRESSION: 
>   - https://github.com/Buttsgroup/imp_predict
>   - https://github.com/Buttsgroup/imp_evaluation
> - hyperparameter_tuning:
>   - https://github.com/Buttsgroup/krr_optimisation
>
> All credit for that code goes to its original authors.

This covers:

- **The entire `mol_translator/` directory**: the `aemol` (AutoEnrich molecule) object, file
  readers/writers, structure utilities, and the `imp_converter` module that builds the atom/pair
  dataframes used as ML input.
- **The IMPRESSION-derived files in `machine_learning/model/`**: `fchl_model.py`,
  `features/fchl_input.py` and the package `__init__.py` files, which keep their original
  copyright and GPL licence headers.
- **The hyperparameter search routines in `data_analysis/hyperparameter_tuning/`**
  (`HPS_gaussian.py`, `HPS_grid.py`, `HPS_random.py`), which also keep their original headers. `DSCRIBEmodel` and `ClassifierModel` follow the same
  structure as the IMPRESSION model classes.

### The bundled `mol_translator` is an old version

**The copy of `mol_translator` in this repository is an old version.** It has not been changed
since it was downloaded, and it is now **far behind the current `mol_translator` package** on
the Butts group GitHub. It is kept here only so the CHAMPS code runs as it did when the paper
was written.

- **Use the bundled copy to run CHAMPS.** The CHAMPS code was written against this version's API.
- **Do not expect the current `mol_translator` to be a drop-in replacement.** Function names,
  module layout and behaviour have probably changed since this copy was taken.
- **For any other project, use the up-to-date package** from the Butts group GitHub, not this copy.

---

## Use the workflow wrappers

> ### ⭐ The workflow wrappers in `data_analysis/workflow.py` are the best and recommended way to run CHAMPS.

The individual modules in `data_analysis/` contain many low-level functions, and they must be
called in the right order with the right intermediate files. The functions in `workflow.py`
chain these steps for you. They handle file naming, run the sanity checks, and write every
intermediate output as `.csv`. **Unless you need to change a specific step, call the workflow
wrappers rather than the underlying functions.**

| Wrapper | What it does |
|---|---|
| `comfile_generation_workflow(dataframe)` | Builds Gaussian 16 inputs for the neutral heterocycle, every C–H deprotonated anion and every brominated intermediate in a VEHICLe dataframe. Writes them to `comfiles/`. |
| `logfile_analysis_workflow(logfiles, comfiles, outname, ...)` | Checks completed Gaussian `.log` files for failed runs and for structures that drifted from the input. Reads energies, calculates acidity and nucleophilicity, optionally calculates relative values and plots the activation map. |
| `impression_input_workflow(neutral_logfiles, fulldata_df, outname)` | Converts neutral-structure log files and the calculated properties into the **atom** and **pair** dataframes used to train and test the ML models. |
| `vehicle_fingerprint_sampling_workflow(vehicle_dataframe, outname, num=100)` | Diversity-samples `num` structures from a VEHICLe dataframe using Tanimoto similarity of fingerprints, starting from SMILES. |
| `logfile_fingerprint_sampling_workflow(neutral_logfiles, training_structures, vehicle_df, outname, fulldata_df, num=100)` | Diversity-samples from structures with completed calculations, and can write the ML inputs for the sample. |

### Typical end-to-end run

```python
import glob
import pandas as pd
from workflow import (
    vehicle_fingerprint_sampling_workflow,
    comfile_generation_workflow,
    logfile_analysis_workflow,
    impression_input_workflow,
)

vehicle = pd.read_csv('../data/dj1.csv')

# 1. (Optional) choose a diverse subset of structures
sample = vehicle_fingerprint_sampling_workflow(vehicle, outname='dj1_sample', num=100)

# 2. Write Gaussian 16 inputs (neutral, anions, bromines) to comfiles/
comfile_generation_workflow(sample)

# ... run the calculations with Gaussian 16 ...

# 3. Check the calculations and compute acidity / nucleophilicity (+ relative values, activation map)
logfiles = glob.glob('logfiles/*.log')
comfiles = glob.glob('comfiles/*.com')
logfile_analysis_workflow(logfiles, comfiles, outname='dj1', plot_map=True)

# 4. Build ML atom/pair dataframes from the neutral structures
neutral_logs = [f for f in logfiles if 'Base_het' in f]
fulldata = pd.read_csv('dj1_fulldata.csv')
atom_df, pair_df = impression_input_workflow(neutral_logs, fulldata, outname='dj1')
```

`logfile_analysis_workflow` writes these files (each prefixed with `outname`):

- `_failed_structures.txt`: calculations that did not end with *Normal termination*
- `_bad_structures.txt`: optimised structures that differ significantly from the input
- `_rawdata.csv`: raw energies read from the log files
- `_fulldata.csv`: absolute acidity and nucleophilicity (kcal/mol) for every site
- `_fulldata_relative.csv`: values relative to pyrazolo[1,5-a]pyrimidine
- `_activation_map.png`: activation map (if `plot_map=True`)

> **Note:** if `calc_check=True`, every log file must have a matching input `.com` file,
> otherwise `logfile_analysis_workflow` raises an error.

---

## Data analysis: acidity and nucleophilicity

Acidity and nucleophilicity come from Gaussian 16 calculations and are handled by
`data_analysis/`, ideally through the [workflow wrappers](#use-the-workflow-wrappers):

1. **Dataset selection**: `VEHICLe_filters.py` filters the VEHICLe database into the subsets used
   in the paper (for example `dj1_filter`, `dj2_filter`, `djr_filter`). `sampling.py` provides
   diversity and random sampling.
2. **Structure and input generation**: `neutral_generation.py`, `anion_generation.py` and
   `bromine_generation.py` turn SMILES into 3D structures and Gaussian `.com` files. One anion
   and one brominated structure are made for each C–H site.
3. **Output checks**: `calculation_check.py` flags failed jobs and optimised structures that drifted
   away from the input geometry.
4. **Property calculation**: `logfile_read.py` reads the energies. `property_calculate.py` then
   calculates:
   - **acidity**: energy of deprotonation at each site (kcal/mol)
   - **nucleophilicity**: electrophile (Br⁺) affinity at each site (kcal/mol)
   - **relative** values against pyrazolo[1,5-a]pyrimidine (`calculate_relative_properties`)
5. **ML input generation**: `impression_input.py` maps each property onto atoms. Acidity goes on
   the C–H hydrogen and nucleophilicity on the C–H carbon, both in the `shift` column of the atom
   dataframe. This file also builds the classifier input dataframes.

---

## Data analysis: donicity

**Donicity is handled separately from acidity and nucleophilicity.** It uses different structures
(N-palladated heterocycles), different software (**ORCA** rather than Gaussian 16) and different
scripts. **The workflow wrappers do not cover it.**

- `nitrogen_palladation.py`: builds the Pd-coordinated structures for each ring nitrogen
- `file_generation.py`: writes ORCA optimisation and frequency inputs (`write_orca_file_for_directed_activation_opt_and_freq`, `frequency_calc_setup`)
- `donicity_data_extraction.py`: checks ORCA output, calculates corrected free energies and Pd exchange free energies (ΔG_exch), and assigns the donicity values to the atom dataframe (`NCS` target)
- `pd_plots.py`: 3D activation maps that include the exchange energies
- `yield_v_properties.py`: compares ΔG and ΔE for the donicity calculations
- `VEHICLe_filters.donicity_filter`: filters and classifies structures by donicity and nucleophilicity limits

---

## Machine learning

### Trained models are included

**The trained CHAMPS models from the paper are in `machine_learning/trained_models/`**, so you can
make predictions without retraining:

| File | Model |
|---|---|
| `CHAMPS-DFT_HCS.pkl` | KRR acidity model, DFT-optimised input structures |
| `CHAMPS-DFT_CCS.pkl` | KRR nucleophilicity model, DFT-optimised input structures |
| `CHAMPS-DFT_NCS.pkl` | KRR donicity model, DFT-optimised input structures |
| `CHAMPS-MM_HCS.pkl` | KRR acidity model, MM-optimised input structures |
| `CHAMPS-MM_CCS.pkl` | KRR nucleophilicity model, MM-optimised input structures |
| `CHAMPS-MM_NCS.pkl` | KRR donicity model, MM-optimised input structures |
| `CHAMPS-classifier_withNHs.pkl` | Site reactivity classifier, trained including N–H heterocycles |
| `CHAMPS-classifier_noNHs.pkl` | Site reactivity classifier, trained excluding N–H heterocycles |

Load them with the `load_model()` method of the matching model class (`DSCRIBEmodel` or
`ClassifierModel`).

### All model architectures are included

All the code needed to rebuild, retrain or extend the models is in `machine_learning/model/`:

- **`DSCRIBEmodel`** (`dscribe_model.py`): the main CHAMPS property model. Kernel Ridge Regression
  on DScribe descriptors (LMBTR, ACSF, SOAP, Coulomb matrix), with train/validation/test splitting,
  Gaussian-process and Bayesian hyperparameter optimisation, and save/load.
- **`ClassifierModel`** (`classifier_model.py`): predicts whether a site is reactive from computed
  or predicted properties and/or structural descriptors. Supports Random Forest, SVC, k-NN, MLP,
  decision tree, SGD and Gaussian process classifiers, with stratified splitting and metrics.
- **`FCHLmodel`** (`fchl_model.py`, from IMPRESSION): KRR on FCHL representations via `qml`.
- **`GDCNN_model`** (`GDCNN.py`): a graph convolutional network in PyTorch Geometric.
- **Features** (`features/`): descriptor generation for each model.

### Making predictions

The full CHAMPS pipeline predicts the three properties with `DSCRIBEmodel`, then passes them to
`ClassifierModel`:

- `total_prediction.py`: from existing atom/pair dataframes (for example DFT structures)
- `molecular_mechanics.py`: from **SMILES only**. It embeds and MMFF-optimises structures with
  RDKit, so no DFT is needed, and then runs the same two-stage prediction.

In outline:

```python
from model.dscribe_model import DSCRIBEmodel
from model.classifier_model import ClassifierModel

ds = DSCRIBEmodel(id='my_predictions')
ds.test_atom_df, ds.test_pair_df = atom_df, pair_df

ds.load_model(model_filename='trained_models/CHAMPS-MM_HCS.pkl')
ds.targetflag = 'HCS'
ds.add_local_many_body_tensor(opt=False, test=True, amols=amols)

for flag in ['HCS', 'CCS', 'NCS']:
    ds.load_model(model_filename=f'trained_models/CHAMPS-MM_{flag}.pkl')
    ds.targetflag = flag
    ds.test_atom_df = ds.assign_preds_from_load(ds.model, ds.test_atom_df, write=False, truth=False)

cm = ClassifierModel(id='my_predictions')
cm.fulldata_atom_df = ds.test_atom_df
cm.fulldata_class_df = class_df
cm.add_dft_to_class_df(cm.fulldata_atom_df, cm.fulldata_class_df, type='predicted_shift')
cm.load_model(model_filename='trained_models/CHAMPS-classifier_withNHs.pkl')
cm.assign_predictions(cm.fulldata_class_df, type='all')
```

> The paths hard-coded in `total_prediction.py` and `molecular_mechanics.py` (for example
> `dj1/classification/...`) point to the original working directories. Change them to point at
> `trained_models/` and your own input data before running.

### Training and evaluation

- `imp_train.py`: `train_and_predict` and `load_and_predict` for the FCHL model
- `hyperparam_opt.py`: cross-validated hyperparameter optimisation
- `top_n_accuracy.py`: checks whether the predicted most reactive site(s) match the DFT order
- `DSCRIBEmodel.krr_train()` / `ClassifierModel.train_model()`: training for the main CHAMPS models

---

## Plots

**All the plotting code is in the `plots` modules:**

- `data_analysis/plots.py`: activation maps (2D, 3D, binary, quadrant), the paper's quadrant
  activation map (`plot_quadrant_activation_map_for_paper`), acidity vs. nucleophilicity, dataset
  counts, bar charts, ΔG vs ΔE comparisons and Gen II activation maps
- `data_analysis/pd_plots.py`: 3D maps that include Pd exchange energies (donicity)
- `machine_learning/plots.py`: DFT vs. ML-predicted values and error distributions

**Where to find the figures from the paper:**

[insert location of paper figures here, e.g. a `figures/` directory, Supporting Information, or a data repository link]

---

## Known issues and caveats

This is research code. It is shared to support the paper and to make the results reproducible,
and it has not been packaged as a polished library.

- **Missing `modules.make_3d_graphs`**: `machine_learning/model/classifier_model.py` imports
  `make_graphs` from `modules.make_3d_graphs`, which is not in this repository. Because the import
  is at the top of the file, `ClassifierModel` (and so `total_prediction.py` and
  `molecular_mechanics.py`) will not import until you either supply the module or remove the
  import. Only the unfinished `ClassifierModel.add_graph_to_class_df()` method uses it.
- **pandas < 2.0 is required** because the code uses `DataFrame.append`.
- **Hard-coded paths**: some scripts (`total_prediction.py`, `molecular_mechanics.py`,
  `yield_v_properties.py`, `regression.py`) contain absolute or project-specific paths to the
  original data. Edit them before running.
- **Windows path separators**: some file-name parsing (for example in
  `logfile_fingerprint_sampling_workflow`) assumes `\` path separators.
- **`imp_input` argument**: in `logfile_analysis_workflow` this argument is currently a placeholder
  and does nothing. Use `impression_input_workflow` instead.

---

## Citation

If you use CHAMPS, please cite:

[insert paper ref here]

If you use the bundled `mol_translator` or IMPRESSION code, please also credit the Butts group:
[insert link here]
