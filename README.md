# Reducing spatial clustering to prevent tuberculosis transmission in a busy Zambian hospital: a modelling study based on person movements, environmental and clinical data

Code and data accompanying the paper *"Reducing spatial clustering to prevent tuberculosis transmission in a busy Zambian hospital: a modelling study based on person movements, environmental and clinical data"* by Nicolas Banholzer, Guy Muula, Fiona Mureithi, Esau Banda, Pascal Bittel, Lavinia Furrer, David Kronthaler, Remo Schmutz, Matthias Egger, Carolyn Bolton, and Lukas Fenner.

The preprint is available at [PMC (medRxiv)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12755300/). The paper is currently under review at *PLOS Medicine*.

**Author affiliations:**
1. Institute of Social and Preventive Medicine, University of Bern, Bern, Switzerland
2. Section of Health Data Science and AI, Department of Public Health, University of Copenhagen, Copenhagen, Denmark
3. Centre for Infectious Disease Research in Zambia (CIDRZ), Lusaka, Zambia
4. Institute for Infectious Diseases, University of Bern, Bern, Switzerland
5. Department of Psychology, University of Zurich, Zurich, Switzerland
6. Division of Clinical Epidemiology, Department of Clinical Research, University of Basel, Basel, Switzerland
7. Department of Infectious Diseases and Hospital Epidemiology, University Hospital Zurich, University of Zurich, Switzerland
8. Population Health Sciences, Bristol Medical School, University of Bristol, Bristol, UK

## Study summary

This study quantifies airborne *Mycobacterium tuberculosis* (Mtb) transmission risk in the outpatient department (OPD) of a hospital in Lusaka, Zambia, using integrated person-tracking, environmental, and clinical data collected over 52 study days (June–August 2024). Two cumulative interventions designed to reduce spatial clustering were evaluated:

1. **First intervention** (July 1–15): optimized waiting-area seating layout with physical distancing measures.
2. **Second intervention** (July 16–29): added a one-way patient flow system.

Transmission risk is estimated using a spatiotemporal extension of the Wells-Riley model based on person movements, as described in [Banholzer et al. (2025), *PLOS Computational Biology*](https://doi.org/10.1371/journal.pcbi.1012823). Infectious quanta emitted by TB patients diffuse radially outward from their tracked positions across the waiting area floor plan (assumed vertically well-mixed), while being removed by ventilation, gravitational settling, and pathogen inactivation. This means that individuals closer to an infectious source are exposed to higher quanta concentrations and thus face a higher infection risk than those further away. This non-uniform mixed model (NUM) captures these proximity effects and is compared against a conventional uniform well-mixed model (UM), which assumes equal exposure everywhere. Details on the model construction, air exchange rate estimation, and effect estimation are provided in the Supplementary Information of the paper.

Key findings: the first intervention reduced spatial clustering by 24% and Mtb transmission risk by 39%; the second intervention reduced clustering by 13% and transmission risk by 21%.

## Requirements

### Python

Python 3.11 or later. Install the required packages:

```bash
pip install numpy pandas scipy shapely h5py numba
```

### R

R 4.3 or later. Install the required packages:

```r
install.packages(c("tidyverse", "lubridate", "brms", "tidybayes",
                    "parallel", "ggpubr", "data.table"))
```

The analysis files also source helper functions for plotting and formatting from [github.com/nbanho/helper](https://github.com/nbanho/helper).

## Repository structure

```
tb_patient_flows/
├── analysis/                  Analysis scripts (R Markdown, Jupyter)
│   ├── clinical.Rmd           Descriptive analysis of clinical TB data
│   ├── environmental.Rmd      CO2, ventilation, and environmental conditions
│   ├── tracking.Rmd           Person movements, spatial clustering, intervention effects
│   ├── tracking.ipynb         Spatial density heatmap visualisation (Python)
│   └── modelling.Rmd          Transmission risk, intervention effects, sensitivity analyses
│
├── preprocessing/             Data preprocessing pipeline (Python, R)
│   ├── read_tracking_flows.py            Convert raw Xovis JSON to per-date CSVs
│   ├── read_tracking_counts.py           Extract entry/exit counts from Xovis JSON
│   ├── prep_tracking_data.py             Filter tracks, annotate spatial zones
│   ├── link_tracking_interrupted_tracks.py  Link interrupted person movements
│   ├── link_clinical_tracking.py         Link TB case IDs to tracking data
│   ├── link_tb_app.R                     Interactive Shiny app for manual TB patient linkage
│   ├── prep_clinical_data.py             Preprocess clinical questionnaire data
│   ├── prep_environmental_data.py        Consolidate CO2 sensor data
│   ├── mapping_dates_interventions.py    Create study phase timeline
│   ├── read_tracking_linked_data.py      Shared utility to read linked tracking data
│   ├── compute_tracking_duration.py      Compute time-in-clinic per person
│   ├── compute_tracking_spatial_density.py   Gini coefficient and Shannon entropy
│   ├── compute_tracking_spatial_density_sensitivity.py  Sensitivity: alternative kernels
│   ├── compute_tracking_spatial_counts.py    Rasterise positions into spatial grid (HDF5)
│   ├── compute_environmental_aer.py      Estimate air exchange rates from CO2
│   ├── get_tracking_positions.py         Assign grid cells and activity status
│   ├── get_tracking_person_features.py   Extract gender and healthcare worker status
│   └── combine_multiple_figures.r        Assemble composite paper figures
│
├── modelling/                 Wells-Riley transmission model (Python, R)
│   ├── run_model.py                  Main simulation entry point
│   ├── spatiotemporal_diffusion.py   2D diffusion–removal PDE solver
│   ├── compute_quanta_exposure.py    Inhaled quanta dose computation
│   ├── create_building_mask.py       Building grid geometry and floor-plan mask
│   ├── modelling_assumptions.r       Sample uncertain model parameters (Monte Carlo)
│   ├── check_convergence.R           Monte Carlo convergence diagnostics
│   └── python_cmd_model_calls        Reference file with example run_model.py invocations
│
├── data-clean/                Preprocessed data (see Data section below)
├── modelling-results/         Simulation output (generated by run_model.py)
└── results/                   Figures and tables (generated by analysis scripts)
```

## Data

### Tracking data (download from OSF)

The processed person-tracking data for all 52 study days will be deposited on OSF upon publication of the paper:

> **[https://doi.org/10.17605/OSF.IO/P3W24](https://doi.org/10.17605/OSF.IO/P3W24)**

Download the CSV files and place them in **`data-clean/tracking/unlinked/`**.

This data is not the raw sensor output. It was derived from the original Xovis JSON exports by converting to CSV format (`preprocessing/read_tracking_flows.py`), filtering to study days and daytime hours (6 AM–6 PM), removing short tracks (<10 seconds), and annotating each position with spatial zone membership (entry areas, seating, TB screening, sputum collection) via `preprocessing/prep_tracking_data.py`. These are light preprocessing steps that do not alter the tracking positions themselves.

The raw Xovis JSON files and the raw entry/exit count data can be shared upon request by contacting the corresponding author.

### Shared in this repository

The following preprocessed data files are included in `data-clean/`:

| Directory | Contents |
|-----------|----------|
| `building/` | Building grid geometry: cell size, binary mask, valid positions, height, volume (`.npy` files) |
| `environmental/` | CO2/temperature/humidity measurements and daily air exchange rate estimates |
| `assumptions/` | 1,000 Monte Carlo draws of quanta generation and removal rates |
| `tracking/linked/` | Linked person movement tracks (52 dates) |
| `tracking/linked-tb/` | TB-relevant linked tracks (52 dates) |
| `tracking/linked-clinical/` | Track-to-TB-case-ID linkage tables (52 dates) |
| `mapping_dates_interventions.csv` | Date-to-study-phase mapping |

**Person movement linkage.** The Xovis sensors assign a new track ID each time a person enters or re-enters the sensor field of view. Since persons may temporarily leave and re-enter (e.g., when visiting a consultation room), `preprocessing/link_tracking_interrupted_tracks.py` links interrupted tracks that likely belong to the same person using spatiotemporal criteria: time gap, spatial distance, and zone membership (seating, walking, TB check area). Linking is applied in 8 rounds of increasing permissiveness. The resulting linkage tables in `tracking/linked/` assign a unified `new_track_id` to each person's full visit. The linkage algorithm is described in detail in the Supplementary Information of the paper.

**TB patient identification.** TB patients were identified using a clinical questionnaire and linked to their tracking data. A research assistant manually matched each TB patient's clinic visit to a tracked person movement using an interactive Shiny application (`preprocessing/link_tb_app.R`). The resulting linkage in `tracking/linked-tb/` marks which tracks belong to TB patients, and `tracking/linked-clinical/` maps track IDs to clinical case IDs. Since this process requires the clinical data, the linkage tables are provided pre-computed. The identification and linkage procedure is described in detail in the Supplementary Information of the paper.

### Clinical data (not shared)

Individual-level clinical data (`data-clean/clinical/tb_cases.csv`) is not shared due to patient privacy. This affects the following scripts, which cannot be re-run from scratch:

- `preprocessing/prep_clinical_data.py`
- `preprocessing/link_clinical_tracking.py`
- `modelling/modelling_assumptions.r`
- `analysis/clinical.Rmd`

However, all outputs produced by these scripts (linked-clinical tracking tables, modelling assumptions) are included in the repository, so the full modelling and analysis pipeline can still be reproduced.

## Reproducing the results

### A. Spatial clustering analysis (crowding and intervention effects)

This pipeline reproduces the spatial clustering (Gini coefficient) analysis and the estimation of intervention effects on crowding patterns.

**Step A1: Link interrupted person movements**

The linkage tables (`data-clean/tracking/linked/` and `data-clean/tracking/linked-tb/`) are already included in the repository. To regenerate them from the unlinked tracking data:

```bash
python preprocessing/link_tracking_interrupted_tracks.py
```

**Step A2: Compute spatial density and occupancy metrics**

```bash
# Compute per-second occupancy, Gini coefficient, and entropy (main analysis)
python preprocessing/compute_tracking_spatial_density.py

# Sensitivity analysis with alternative kernels (Gaussian vs exponential)
python preprocessing/compute_tracking_spatial_density_sensitivity.py

# Compute time-in-clinic per person
python preprocessing/compute_tracking_duration.py

# Compute spatial density grid for heatmap visualisation
python preprocessing/compute_tracking_spatial_counts.py
```

**Step A3: Run the spatial clustering analysis**

```bash
Rscript -e 'rmarkdown::render("analysis/tracking.Rmd")'
```

This fits Bayesian Beta regression models of the Gini coefficient on study phase and weekday, estimates intervention effects on spatial clustering, and runs sensitivity analyses with alternative kernels and metrics.

### B. Transmission modelling (infection risk and intervention effects)

This pipeline reproduces the Wells-Riley spatiotemporal transmission risk simulations and the estimation of intervention effects on Mtb infections. The model construction, parameter assumptions, and effect estimation approach are described in detail in the Supplementary Information of the paper.

**Step B1: Generate position inputs for the model**

```bash
python preprocessing/get_tracking_positions.py
```

This assigns grid cell indices and walking/sitting activity status to each tracking position, and separates TB and non-TB patient positions.

**Step B2: Run simulations**

```bash
cd modelling

# Main analysis: all TB cases equally infectious
python run_model.py --name "uncertain_all_equal_infectious" --date "all" --sim "(1,100)"

# Sensitivity scenario 1: confirmed TB 8x more infectious than presumptive
python run_model.py --name "uncertain_confirmed_more_infectious" --date "all" --sim "(1,100)" --quanta_mult "(1.0, 8.0, 1.0)"

# Sensitivity scenario 2: HIV-positive individuals more infectious
python run_model.py --name "uncertain_hiv_more_infectious" --date "all" --sim "(1,100)" --quanta_mult "(0.67, 1.0, 1.0)"

# Sensitivity scenario 3: all presumptive TB cases are infectious
python run_model.py --name "uncertain_all_infectious" --date "all" --sim "(1,100)" --all_infectious

cd ..
```

**Step B3: Extract person features and run analysis**

```bash
# Extract gender and healthcare worker status for stratified analysis
python preprocessing/get_tracking_person_features.py

# Run the modelling analysis
Rscript -e 'rmarkdown::render("analysis/modelling.Rmd")'
```

This estimates daily expected Mtb infections, computes intervention effects via Bayesian regression across Monte Carlo draws, predicts counterfactual prevented infections, and runs sensitivity analyses.

### C. Environmental analysis

The environmental sensor data and air exchange rate estimates are already included in `data-clean/environmental/`. The AER estimation method (transient mass balance model fitted to CO2 decay curves) is described in detail in the Supplementary Information of the paper. To reproduce the AER estimation and the environmental analysis:

```bash
# Recompute air exchange rates from CO2 data (optional; output already included)
python preprocessing/compute_environmental_aer.py

# Run the environmental analysis
Rscript -e 'rmarkdown::render("analysis/environmental.Rmd")'
```

### Full pipeline from scratch

To reproduce the complete preprocessing from the unlinked tracking data (already included in the repository or available on [OSF](https://doi.org/10.17605/OSF.IO/P3W24)), run Steps A1–A3, then B1–B3, then C above in order.

## Citation

If you use this code or data, please cite the preprint:

> Banholzer N, Muula G, Mureithi F, Banda E, Bittel P, Furrer L, Kronthaler D, Schmutz R, Egger M, Bolton C, Fenner L. "Reducing spatial clustering to prevent tuberculosis transmission in a busy Zambian hospital: a modelling study based on person movements, environmental and clinical data." *medRxiv* (2025). DOI: [10.64898/2025.12.24.25342956](https://doi.org/10.64898/2025.12.24.25342956)

If you use only the data, you may alternatively cite the OSF repository:

> Banholzer N et al. "Data: Reducing spatial clustering to prevent tuberculosis transmission in a busy Zambian hospital." *OSF* (2025). DOI: [10.17605/OSF.IO/P3W24](https://doi.org/10.17605/OSF.IO/P3W24)

For the spatiotemporal Wells-Riley model methodology, please also cite:

> Banholzer N, Middelkoop K, Leukes J, Weingartner E, Schmutz R, Zürcher K, Egger M, Wood R, Fenner L. "Estimating *Mycobacterium tuberculosis* transmission in a South African clinic: Spatiotemporal model based on person movements." *PLOS Computational Biology* (2025). DOI: [10.1371/journal.pcbi.1012823](https://doi.org/10.1371/journal.pcbi.1012823)

## Data use agreement

The anonymized data in this repository and on OSF were collected at the Centre for Infectious Disease Research in Zambia (CIDRZ) with approval from the National Health Research Authority (NHRA) of Zambia and the University of Zambia Biomedical Research Ethics Committee (UNZABREC). Public release of these data was approved by the NHRA (Ref: NHRA/DP/03/2026).

By using these data, you agree to the following terms:

- **No re-identification.** You shall not attempt to re-identify any individual in the dataset.
- **Acknowledgment.** You must acknowledge CIDRZ and the NHRA as the source and approving authority for the data in any publication or presentation.
- **Non-commercial use.** The data may only be used for non-commercial, scientific research purposes.
- **Publication disclaimer.** Any publications arising from these data must include the following disclaimer: *"The data used in this study were collected with approval from the National Health Research Authority (NHRA) of Zambia and the University of Zambia Biomedical Research Ethics Committee (UNZABREC). The interpretation and conclusions are solely those of the authors."*

## License

- **Code** (all files in `preprocessing/`, `modelling/`, and `analysis/`): [MIT License](https://opensource.org/licenses/MIT).
- **Data** (all files in `data-clean/` and on OSF): [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (Attribution-NonCommercial), subject to the data use agreement above.

## Acknowledgments

We thank the National Health Research Authority (NHRA) of Zambia and the University of Zambia Biomedical Research Ethics Committee (UNZABREC) for ethical approval, and the Centre for Infectious Disease Research in Zambia (CIDRZ) for facilitating data collection.
