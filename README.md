# A Predictive–Causal Machine Learning Framework for Public Expenditure Execution in the DR Congo

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red)
![License](https://img.shields.io/badge/License-Academic-green)

Replication code and data for the paper *"A Predictive–Causal Machine Learning Framework for Public Expenditure Execution: Evidence From the Democratic Republic of Congo (1996–2023)"* (Nzazi et al., IEEE Access).

## Description

This repository provides the full pipeline behind the article: an original panel of budget-execution data for the Democratic Republic of Congo (DRC) and Angola, together with the predictive and causal machine learning code used to analyse it. The framework separates *prediction* (which variables forecast execution) from *causal identification* (which variables a reform can act upon), combining ensemble forecasting, regime classification, SHAP interpretation, Double Machine Learning (DML) and Causal Forest estimation.

An interactive Streamlit application exposes the data and results through a browsable interface.

**Repository:** https://github.com/BoazNzazi/Causal-Predictive-ML-Budget-DRC

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data](#data)
- [Running the Application](#running-the-application)
- [Application Modules](#application-modules)
- [Reproducibility](#reproducibility)
- [Dependencies](#dependencies)
- [Authors](#authors)
- [Citation](#citation)
- [License](#license)

## Project Structure

```
Causal-Predictive-ML-Budget-DRC/
│
├── Main.py                              # Entry point — Streamlit application
├── FINAL_WITH_ALL_INDICATEURS.xlsx      # Consolidated dataset (all indicators)
├── requirements                         # Pinned Python dependencies
├── .gitignore
│
├── sections/                            # Application modules (one per tab)
│   ├── accueil.py                       # Home / Welcome page
│   ├── data.py                          # Data exploration & preview
│   ├── visualisation.py                 # Charts and dashboards
│   ├── analyse.py                       # Statistical & causal analysis
│   ├── nlp.py                           # Natural Language Processing module
│   └── apropos.py                       # About / project information
│
├── env/                                 # (Virtual environment — not tracked)
└── monenviron/                          # (Virtual environment — not tracked)
```

## Requirements

| Tool   | Recommended version | Notes                                   |
|--------|---------------------|-----------------------------------------|
| Python | 3.9 or higher       | Tested on 3.9, 3.10 and 3.11            |
| pip    | 23+                 | `python -m pip install --upgrade pip`   |
| Git    | any recent version  | to clone the repository                 |

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/BoazNzazi/Causal-Predictive-ML-Budget-DRC.git
cd Causal-Predictive-ML-Budget-DRC
```

### Step 2 — Create and activate a virtual environment (recommended)

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies

```bash
pip install -r requirements
```

## Data

The dataset used by the application is:

| File                              | Format | Description                                                        |
|-----------------------------------|--------|--------------------------------------------------------------------|
| `FINAL_WITH_ALL_INDICATEURS.xlsx` | Excel  | Consolidated dataset with all indicators of public expenditure in the DRC and Angola |

The file is loaded automatically at startup in `Main.py`:

```python
df = pd.read_excel('FINAL_WITH_ALL_INDICATEURS.xlsx')
```

Make sure the file is present in the project root before launching the application.

The panel is reconstructed from three Banque Centrale du Congo (BCC) annual reports (2005, 2014, 2023), harmonised into 28 stable functional categories over 1996–2023, and integrated with IMF and World Bank macroeconomic and governance indicators. An independent Angolan panel (2011–2022) is included for external validation.

## Running the Application

From the project root (with the virtual environment activated):

```bash
streamlit run Main.py
```

The application opens automatically in your default web browser at:

```
http://localhost:8501
```

## Application Modules

The application is organised into six navigation sections accessible from the sidebar:

| Section        | File                       | Description                                                  |
|----------------|----------------------------|--------------------------------------------------------------|
| Home           | `sections/accueil.py`      | Landing page, project overview                               |
| Data           | `sections/data.py`         | Raw exploration, filtering and data preview                  |
| Visualisation  | `sections/visualisation.py`| Interactive charts, dashboards and spatial maps              |
| Analysis       | `sections/analyse.py`      | Statistical analysis, correlations and causal estimation     |
| NLP            | `sections/nlp.py`          | Natural Language Processing applied to public-finance data   |
| About          | `sections/apropos.py`      | Project information, methodology and team                    |

## Reproducibility

This repository is released to ensure the full reproducibility of the results presented in the associated research article.

To reproduce all results:

1. Clone the repository (see [Installation](#installation)).
2. Install the exact dependency versions listed in `requirements` (all versions are pinned).
3. Launch the application with `streamlit run Main.py`.
4. Navigate through each section to reproduce the figures, tables and analyses described in the article.

### Environment snapshot

All dependencies are pinned in the `requirements` file. Key libraries:

```
streamlit==1.38.0
pandas==2.2.2
numpy==2.1.1
matplotlib==3.9.2
seaborn==0.13.2
openpyxl==3.1.5
altair==5.4.1
pyarrow==17.0.0
```

The full list is provided in the `requirements` file at the repository root.

### Python version

```
Python 3.9+  (tested on 3.9, 3.10, 3.11)
```

## Dependencies

Install all dependencies with:

```bash
pip install -r requirements
```

Main libraries used:

- **Streamlit** — interactive web application framework
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical computing
- **Matplotlib / Seaborn** — static visualisations
- **Altair** — declarative interactive visualisations
- **OpenPyXL** — reading and writing Excel files
- **PyArrow** — columnar data processing

## Authors

- **Boaz N. Nzazi** — M.Sc. in Intelligent Systems — GitHub: [@BoazNzazi](https://github.com/BoazNzazi)
- **Jirince K. Biaba**
- **Christian M. Mulomba**
- **Nathanaël M. Kasoro**
- **Selain K. Kasereka** — Ph.D. in Mathematics and Computer Science — GitHub: [@sedjokas](https://github.com/sedjokas)

ABIL Research Center, Kinshasa, Democratic Republic of the Congo.

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{nzazi2026predictive,
  author  = {Nzazi, Boaz N. and Biaba, Jirince K. and Mulomba, Christian M.
             and Kasoro, Nathana{\"e}l M. and Kasereka, Selain K.},
  title   = {A Predictive--Causal Machine Learning Framework for Public
             Expenditure Execution: Evidence From the Democratic Republic
             of Congo (1996--2023)},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Code and data: \url{https://github.com/BoazNzazi/Causal-Predictive-ML-Budget-DRC}}
}
```

## License

This project is distributed for research and reproducibility purposes. Please contact the authors for any reuse beyond academic citation.

---

*Last updated: July 2026*
