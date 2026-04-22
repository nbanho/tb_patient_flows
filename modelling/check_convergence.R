# ============================================================================
# Check Monte Carlo convergence of simulation results.
#
# Plots cumulative quantiles of expected infections across simulations for
# one example date, verifying that the median and credible intervals stabilize
# as the number of simulations increases.
#
# Reads from:  modelling-results/uncertain_all_equal_infectious/2024-06-26/
# Writes to:   results/modelling_convergence.png
# ============================================================================

library(tidyverse)

source("https://raw.githubusercontent.com/nbanho/helper/refs/heads/main/R/plotting.R")

# files for one day
sim_files <- list.files(
  "modelling-results/uncertain_all_equal_infectious/2024-06-26/",
  pattern = ".csv$", full.names = TRUE
)

# compute expected infections per simulation
sim_daily_inf <- map_dfr(sim_files, function(file) {
  read_csv(file) %>%
    mutate(
      sim = as.integer(str_extract(basename(file), "(?<=_sim_)\\d+")),
      risk = 1 - exp(-conc_diffusion)
    ) %>%
    group_by(sim) %>%
    summarise(exp_inf = sum(risk)) %>%
    ungroup()
})

# plot convergence
sim_daily_inf %>%
  ggplot(aes(x = sim, y = exp_inf)) +
  geom_point() +
  geom_segment(aes(yend = 0, xend = sim))
# Running quantile: compute quantile over the first 1..n elements
cumquantile <- function(x, p = .5) {
  purrr::map_dbl(seq_along(x), ~ quantile(x[1:.x], p))
}
conv_pl <- sim_daily_inf %>%
  arrange(sim) %>%
  mutate(
    median = cumquantile(exp_inf, .5),
    lower = cumquantile(exp_inf, .25),
    upper = cumquantile(exp_inf, .75),
    lower_80 = cumquantile(exp_inf, .1),
    upper_80 = cumquantile(exp_inf, .9),
    lower_95 = cumquantile(exp_inf, .025),
    upper_95 = cumquantile(exp_inf, .975)
  ) %>%
  ggplot(aes(x = sim)) +
  geom_line(aes(y = median)) +
  geom_line(aes(y = lower, color = "50%-CrI"), linetype = "dashed") +
  geom_line(aes(y = upper, color = "50%-CrI"), linetype = "dashed") +
  geom_line(aes(y = lower_80, color = "80%-CrI"), linetype = "dashed") +
  geom_line(aes(y = upper_80, color = "80%-CrI"), linetype = "dashed") +
  geom_line(aes(y = lower_95, color = "95%-CrI"), linetype = "dashed") +
  geom_line(aes(y = upper_95, color = "95%-CrI"), linetype = "dashed") +
  scale_y_log10(name = "Estimated number of Mtb infections") +
  scale_x_continuous(
    name = "Number of simulations",
    expand = expansion(mult = c(0, 0.01)), limits = c(1, NA)
  ) +
  scale_color_brewer(name = "Credible interval", type = "qual", palette = 1) +
  theme_custom() +
  theme(legend.title = element_blank())
save_plot(
  conv_pl,
  "results/modelling_convergence.png",
  w = 16, h = 10
)
