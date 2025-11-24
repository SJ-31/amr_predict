suppressMessages({
  library(tidyverse)
  library(here)
  library(glue)
  library(logger)
})
log_path <- here("data", "meta", "biosample_mapping.log")
if (file.exists(log_path)) {
  file.remove(log_path)
}
lf <- appender_file(log_path)

log_layout(layout = layout_blank, namespace = "tb_view")
log_formatter(formatter = formatter_pander, namespace = "tb_view")
log_appender(lf)
log_appender(lf, namespace = "tb_view")

bsample_map <- read_csv(here(
  "data",
  "meta",
  "biosample_mapping_2025-11-21.csv"
)) |>
  filter(!is.na(BioSample))

ast <- read_tsv(here("data", "raw", "asts.tsv"))

standard <- "CLSI"

joined <- inner_join(
  bsample_map,
  ast,
  by = join_by(x$BioSample == y$`#BioSample`)
) |>
  filter(!is.na(`MIC (mg/L)`) & `Testing standard` == standard) |>
  rename(MIC = "MIC (mg/L)") # MIC measured in (mg/L)
# TODO: how to handle "Measurement sign"?

# Unique samples
utb <- joined |> distinct(BioSample, .keep_all = TRUE)

# Number of samples with data
n_samples <- length(utb$BioSample)
log_info("Number of samples with data: {n_samples}")

mic_values <- joined |>
  select(BioSample, ScientificName, MIC, Antibiotic) |>
  pivot_wider(
    names_from = Antibiotic,
    values_from = MIC,
    id_cols = c(BioSample, ScientificName),
    values_fn = first,
  )

amr_phenotype <- joined |>
  select(BioSample, `Resistance phenotype`, Antibiotic) |>
  pivot_wider(
    names_from = Antibiotic,
    values_from = `Resistance phenotype`,
    id_cols = BioSample,
    values_fn = first,
  ) |>
  rename_with(\(x) paste0(x, "_class"))

final <- inner_join(
  mic_values,
  amr_phenotype,
  by = join_by(x$BioSample == y$BioSample_class)
)

null_description <- colSums(is.na(select(
  mic_values,
  -c(BioSample, ScientificName)
))) |>
  sort() |>
  as.data.frame() |>
  `colnames<-`("null_count") |>
  rownames_to_column(var = "antibiotic") |>
  mutate(null_percent = null_count / nrow(mic_values))

table(mic_values$ScientificName) |>
  sort(decreasing = TRUE) |>
  as.data.frame() |>
  `colnames<-`(c("taxon", "count")) |>
  mutate(percentage = count / nrow(mic_values)) |>
  write_tsv(here("data", "meta", "ncbi_all_taxon_counts.tsv"))

write_tsv(final, here("data", "meta", "ncbi_all_samples.tsv"))

write_tsv(
  null_description,
  here("data", "meta", "ncbi_all_antibiotic_null_counts.tsv")
)
