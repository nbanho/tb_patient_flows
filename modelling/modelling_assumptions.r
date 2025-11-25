# libraries
library(tidyverse)

# seed
set.seed(12345)

# number of samples
n_samples <- 1e3

# potentially infectious tracks
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

# quanta generation rates
quanta_rates <- track_ids %>%
  mutate(sample = list(1:n_samples)) %>%
  unnest(cols = c(sample)) %>%
  mutate(
    waiting_rate = rlnorm(n = n(), meanlog = 0.07162337, sdlog = 2.99399),
    walking_rate = rlnorm(n = n(), meanlog = 1.035785, sdlog = 2.99252)
  )

# subsample of presumptive is not infectious
prob_infectious <- tibble(
  sample = 1:n_samples,
  prob = runif(n_samples, min = 0.1, max = 0.3)
)
quanta_rates <- quanta_rates %>%
  left_join(prob_infectious, by = "sample") %>%
  group_by(sample) %>%
  mutate(infectious = if_else(tb == 1, 1, rbinom(n(), 1, prob))) %>%
  ungroup() %>%
  mutate(
    waiting_rate = waiting_rate * infectious,
    walking_rate = walking_rate * infectious
  )

# save quanta generation rates per track id
write.csv(
  quanta_rates,
  "data-clean/assumptions/quanta_generation_rates.csv",
  row.names = FALSE
)

# quanta removal rates
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
