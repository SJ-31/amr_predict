library(tidyverse)
library(curl)
library(glue)
## library(taxizedb)
library(here)

source(here("src", "R", "utils.R"))

## * GOA

# GOA statistics file downloaded from
# https://www.ebi.ac.uk/inc/drupal/goa/proteomes_release.html (04-Mar-2026 release) 2026-04-10
# and parsed into the table "goa_stats.tsv"

outdir <- here("data", "remote", "datasets/2026-04-10_GOA")
completion_marker <- here(outdir, "complete.txt")
other_accs <- read_lines(here("data", "remote", "datasets/acc_list.txt"))

if (!file.exists(completion_marker)) {
  url <- "https://ftp.ebi.ac.uk/pub/databases/GO/goa/proteomes"
  ncbi <- src_ncbi(here("data", "remote", "cache", "taxizedb", "NCBI.sql"))
  stats <- read_tsv(here(
    "data",
    "meta",
    "2026-04-10_goa_stats.tsv"
  )) |>
    mutate(`Percentage coverage` = str_remove(`Percentage coverage`, "%")) |>
    arrange(desc(`Percentage coverage`))

  selected <- stats |>
    head(1000)

  mapped_ids <- ncbi_taxid2rank(selected$`Tax ID`, "phylum")

  selected <- selected |> mutate(Phylum = mapped_ids[as.character(`Tax ID`)])
  write_lines(paste0(url, "/", selected$File), here(outdir, "file_list.txt"))
}

# Only accept proteins with at least a few high-quality annotations

accepted_ecs <- c(
  "EXP",
  "IDA",
  "IPI",
  "IMP",
  "IGI",
  "IEP",
  "HTP",
  "HDA",
  "HMP",
  "HGI",
  "HEP"
)

chosen_goa <- read_existing(here(outdir, "go_meta.tsv"), \(file) {
  all_goa <- lapply(
    list.files(outdir, pattern = ".goa", full.names = TRUE),
    \(x) suppressMessages(read_goa(x))
  ) |>
    data.table::rbindlist() |>
    as_tibble()
  selected <- all_goa |>
    filter(Evidence_Code %in% accepted_ecs) |>
    pull(DB_Object_ID) |>
    unique()
  message(glue("{length(selected)} proteins kept after filtering"))
  final <- all_goa |> filter(DB_Object_ID %in% selected)
  write_tsv(final, file)
  final
})

## confidence <- read_existing(
##   here("data", "meta", "uniprot_confidence.tsv"),
##   \(file) {
##     tb <- get_uniprot_confidence(chosen_goa)
##     write_tsv(tb, file)
##     tb
##   },
##   read_tsv
## )
