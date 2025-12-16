suppressMessages({
  library(here)
  library(tidyverse)
  library(glue)
  source(here("src", "R", "utils.R"))
})
META <- here("data", "meta")

aware <- read_tsv(here("data", "meta", "who_aware.tsv"))
nulls <- read_tsv(here("data", "meta", "ncbi_all_antibiotic_null_counts.tsv"))
env <- yaml::read_yaml(here("config", "ast_browser_pred.yaml"))

DATE <- Sys.Date()
OUT <- here("results", "figures", "eda")
# Provide more metrics and visualizations for the dataset obtained by
# `format_ncbi_metadata.R` (the file "ast_subsampled.tsv")

## * Utilities

FIT_TO <- seq(-8, 10)
smoothen_log2 <- function(x) {
  if (is.na(x)) {
    return(x)
  }
  lg <- log2(x)
  rounded <- round(lg)
  if (lg == rounded) {
    x
  } else {
    nearest <- which.min(abs(FIT_TO - lg))
    2^FIT_TO[nearest]
  }
}

plot_helper <- function(plot, name, width = 15, height = 10) {
  ggsave(
    here(OUT, glue("{name}-{DATE}.png")),
    plot = plot,
    height = height,
    width = width
  )
}

save_tb_helper <- function(x, name) {
  write_tsv(x, here(OUT, glue("{name}-{DATE}.tsv")))
}

## * Load data

aware <- aware %>%
  mutate(
    name = str_remove(
      str_replace(str_to_lower(name), " \\+ ", "-"),
      "\\(.*\\)"
    ),
    name = case_match(
      name,
      "sulfamethoxazole-trimethoprim" ~ "trimethoprim-sulfamethoxazole",
      .default = name
    )
  ) |>
  distinct()

joined <- left_join(aware, nulls, by = join_by(x$name == y$antibiotic))

pdata <- read_tsv(here("data", "meta", "ncbi_all_samples.tsv")) |>
  select(all_of(c(
    "BioSample",
    env$pool_embeddings$obs_keep
  )))

acols <- colnames(pdata)[-1] |> discard(\(x) str_detect(x, "_class"))

tb <- read_tsv(here(META, "ast_subsampled.tsv")) |>
  inner_join(pdata, by = join_by(BioSample))

na_description <- describe_na(tb[, acols], "antibiotic")

# Genera with at least 50 samples, don't plot the minorities
genera_show <- table(tb$genus) |>
  discard(\(x) x < 50) |>
  names()

## * Visualize
# %%

show_classes <- tb |>
  pivot_longer(
    cols = ends_with("_class"),
    names_to = "antibiotic",
    values_to = "class"
  ) |>
  mutate(
    antibiotic = str_remove(antibiotic, "_class"),
    class = case_match(class, "not defined" ~ NA, .default = class)
  )

show_vals <- tb |>
  pivot_longer(
    cols = acols,
    names_to = "antibiotic",
    values_to = "mic"
  ) |>
  mutate(mic_smooth = map_dbl(mic, smoothen_log2))

vals_by_project <- show_vals |>
  ggplot(aes(x = mic_smooth, fill = antibiotic)) +
  geom_histogram() +
  scale_y_continuous(transform = "log2") +
  facet_wrap(~umbrella_project) +
  xlab("MIC (mg/L,smoothened)")

vals_by_genus <- show_vals |>
  filter(genus %in% genera_show & !is.na(mic_smooth)) |>
  ggplot(aes(x = mic_smooth, fill = antibiotic)) +
  geom_histogram() +
  scale_y_continuous(transform = "log2") +
  facet_wrap(~genus) +
  xlab("MIC (mg/L, smoothened)")

class_by_project <- show_classes |>
  filter(!is.na(class)) |>
  ggplot(aes(x = class, fill = umbrella_project)) +
  geom_bar() +
  facet_wrap(~antibiotic)

class_by_genus <- show_classes |>
  filter(genus %in% genera_show & !is.na(class)) |>
  ggplot(aes(x = class, fill = genus)) +
  geom_bar() +
  facet_wrap(~antibiotic)


confounding_score_multi(
  tb,
  c("interest_group", "genus", "umbrella_project", "collection_year")
)

plot_helper(vals_by_project, "mic_umbrella_project")
plot_helper(vals_by_genus, "mic_genus", width = 20)
plot_helper(class_by_project, "amr_class_umbrella_project")
plot_helper(class_by_genus, "amr_class_genus")

# TODO: do a write-up of these figures once you finalize the dataset i.e. figure out
# what genomes are available

cfs <- confounding_score_multi(tb, c("genus", paste0(acols, "_class")))

# Determine what genera would be valid to use for splitting in a generalization evaluation for the specific antibiotic
# The genus has to have enough "resistant" and "susceptible"

# TODO: could you still include rare genera in a test set?
# So long as you acknowledge that we won't know how the model can classify the missing labels for that genus
min_resistant <- 10
min_susceptible <- 50
valid_for_splits <- cfs$tables |>
  keep_at(\(x) str_detect(x, "genus")) |>
  lmap(\(x) {
    am <- names(x) |>
      str_remove("^genus_") |>
      str_remove("_class$")
    table2tb(x[[1]], "genus") |>
      filter(resistant > min_resistant & susceptible > min_susceptible) |>
      mutate(antibiotic = am)
  }) |>
  bind_rows()

save_tb_helper(valid_for_splits, "amr_classes_genus_counts")
