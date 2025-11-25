# Reducing spatial clustering to prevent tuberculosis transmission in a busy Zambian hospital: a modelling study based on person movements, environmental and clinical data

This repository contains the code accompanying the paper "Reducing spatial clustering to prevent tuberculosis transmission in a busy Zambian hospital: a modelling study based on person movements, environmental and clinical data" by Banholzer et al.

The repository is structed as follows:

1. **analysis**: R Markdown and Jupyter Notebooks for descriptive analysis of the different types of data (clinical, environmental and person-tracking data), analysis of the modelling results, and estimation of the intervention effects.
2. *data-check*: Internally used code files to inspect and check the raw and cleaned data.
3. *data-clean*: Empty, all data will be made available upon publication.
4. *data-raw*: Empty, all data will be made available upon publication.
5. *doc*: Empty, placeholder for text files for article writing, publication and study documentation.
6. *modelling-results*: Empty, folder where the simulation results from modelling are stored.
7. **modelling**: Python code files for the spatiotemporal extension of the Wells-Riley model to estimate the transmission risk of Mycobacterium tuberculosis (Mtb) in the Zambian hospital.
8. **preprocessing:** R and Python code files to preprocess the raw data, prepare the input data for modelling, and compute spatial density and ventilation measures.
9. *results*: Empty, placeholder for descriptive, modelling and estimation result figures.

The important code files are as follows:

- **analysis/modelling.rmd:** Contains code to analyse the simulation results for Mtb transmission risk and performs estimation of the IPC intervention effects on transmission.
- **analysis/tracking.rmd:**: Contains code to analyse the person tracking data and performs estimation of the IPC intervention effects on the spatial clustering of people.
- **modelling/run_model.py:** Runs the spatiotemporal Wells-Riley model for the Zambian hospital.
- **modelling/spatiotemporal_diffusion.py:** Python implementation of the general spatiotemporal Wells-Riley model used in modelling/run_model.py.
- **preprocessing/compute_spatial_tracking_density.py**: Computes the Gini coefficient (a measure for the spatial clustering of people) based on smoothed person counts in the hospital's waiting area.
- **preprocessing/compute_environmental_aer.py**: Computes the outdoor air exchange rate based on CO_2 measurements and hospital occupancy.
- **preprocessing/link_tb_app.R**: R Shiny App to manually identify person movements probably belonging to TB patients and linking their possibly interrupted movements.
