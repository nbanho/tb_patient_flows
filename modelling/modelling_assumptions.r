# ============================================================================
# Sample uncertain model parameters for Monte Carlo TB transmission simulation.
#
# Generates 1000 draws of:
#   - Quanta generation rates (waiting and walking) from log-normal priors
#   - Infectiousness status for presumptive TB cases (Bernoulli with
#     uniform probability 10-30%)
#   - Quanta removal rates (settling and inactivation) from log-normal priors
#
# Reads from:
#   data-clean/tracking/linked-clinical/  (track-to-patient linkage)
#   data-clean/clinical/tb_cases.csv      (TB test results and HIV status)
# Writes to:
#   data-clean/assumptions/quanta_generation_rates.csv
#   data-clean/assumptions/quanta_removal_rates.csv
# ============================================================================

library(tidyverse)

set.seed(12345)

n_samples <- 1e3  # number of Monte Carlo draws

# Potentially infectious tracks: linked tracking IDs with clinical records
track_ids <- lapply(
  list.files("data-clean/tracking/linked-clinical/",
    full.names = TRUE, pattern = "*.csv", recursive = TRUE
  ),
  function(x) {
    read.csv(x) %>%
      mutate(date = gsub(".csv", "", basename(x), fixed = TRUE))
  }
)
track_ids <- do.call(rbind, track_ids) %>%
  filter(!is.na(clinic_id)) %>%
  dplyr::select(date, new_track_id, clinic_id) %>%
  group_by(date, new_track_id) %>%
  slice(1) %>%
  ungroup()

# add TB test and HIV status
clinical_df <- read.csv("data-clean/clinical/tb_cases.csv") %>%
  filter(tb_status %in% c("infectious", "presumptive")) %>%
  mutate(
    hiv = if_else(tolower(hiv_status) == "positive", 1, 0),
    tb = if_else(tolower(tb_status) == "infectious", 1, 0),
    across(c("hiv", "tb"), ~ as.integer(replace_na(.x, 0)))
  ) %>%
  dplyr::select(clinic_id, hiv, tb)
track_ids <- track_ids %>%
  left_join(clinical_df, by = "clinic_id")

# Quanta generation rates (1/h) sampled from log-normal distributions.
# Parameters derived from Nguyen et al. (2024) meta-analysis of TB quanta
# emission studies:
#   Waiting (sedentary): meanlog=0.072, sdlog=2.994
#   Walking (light activity): meanlog=1.036, sdlog=2.993
quanta_rates <- track_ids %>%
  mutate(sample = list(1:n_samples)) %>%
  unnest(cols = c(sample)) %>%
  mutate(
    waiting_rate = rlnorm(n = n(), meanlog = 0.07162337, sdlog = 2.99399),
    walking_rate = rlnorm(n = n(), meanlog = 1.035785, sdlog = 2.99252)
  )

# Infectiousness probability for presumptive TB cases.
# Confirmed cases (tb==1) are always infectious; presumptive cases (tb==0)
# are infectious with probability drawn from Uniform(0.1, 0.3), reflecting
# the estimated 10-30% positivity rate among presumptive cases.
prob_infectious <- tibble(
  sample = 1:n_samples,
  prob = runif(n_samples, min = 0.1, max = 0.3)
)
quanta_rates <- quanta_rates %>%
  left_join(prob_infectious, by = "sample") %>%
  group_by(sample) %>%
  mutate(infectious = if_else(tb == 1, 1, rbinom(n(), 1, prob))) %>%
  ungroup()

# save quanta generation rates per track id
write.csv(
  quanta_rates,
  "data-clean/assumptions/quanta_generation_rates.csv",
  row.names = FALSE
)

# Quanta removal rates (1/h) sampled from log-normal distributions.
# Settling rate: gravitational deposition of droplet nuclei (Fennelly, 2020)
# Inactivation rate: Mtb viability loss in aerosol (Dharmadhikari et al., 2012)
settling <- rlnorm(n_samples, meanlog = 0.3624846, sdlog = 0.517269)
inactivation <- rlnorm(n_samples, meanlog = 0.0008491922, sdlog = 0.9993368)
quanta_removal_rates <- tibble(
  sample = 1:n_samples,
  settling = settling,
  inactivation = inactivation,
  total = settling + inactivation
)

write.csv(
  quanta_removal_rates,
  "data-clean/assumptions/quanta_removal_rates.csv",
  row.names = FALSE
)
