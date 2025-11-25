# libraries
library(tidyverse)

# helper functions
source("https://raw.githubusercontent.com/nbanho/helper/refs/heads/main/R/plotting.R")
source("https://raw.githubusercontent.com/nbanho/helper/refs/heads/main/R/formatting.R")

# data overview
clinical_data_pl <- readRDS("results/clinical_tb_cases_over_time.rds") +
  ggtitle("A")
track_data_pl <- readRDS("results/tracking_occupancy_over_time.rds") +
  ggtitle("B")
env_data_co2_pl <- readRDS("results/environmental_co2_over_time.rds") +
  ggtitle("C")
data_pl <- clinical_data_pl + track_data_pl + env_data_co2_pl +
  plot_layout(ncol = 1, heights = c(5, 9.5, 7.5))
save_plot(data_pl, "results/modelled_data.png", w = 16, h = 22)

# modelling effects
effects_density_pl <- readRDS("results/modelling_estimated_reduction_in_spatial_density.rds") +
  ggtitle("A")
effects_infections_pl <- readRDS("results/modelling_estimated_reduction_in_infections.rds") +
  ggtitle("B")
effects_pl <- effects_density_pl + effects_infections_pl +
  plot_layout(ncol = 1, heights = c(7.5, 7.5))
save_plot(effects_pl, "results/modelled_effects.png", w = 16, h = 15)
