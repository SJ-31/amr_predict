library(here)
library(tidyverse)
library(glue)

metadata <- read_tsv(here("data", "meta", "ast_subsampled.tsv"))
ast_dir <- here("data", "remote", "ast_browser")

data_dir <- "~/amr_predict/data/remote/ast_browser/raw/"
directories <- list.dirs(ast_dir, recursive = FALSE, full.names = FALSE) |>
  keep(\(x) x %in% metadata$Run)

run2biosample <- setNames(metadata$BioSample, metadata$Run)

tb <- tibble(
  ID = run2biosample[directories],
  R1 = paste0(data_dir, directories, "_1.fastq.gz"),
  R2 = paste0(data_dir, directories, "_2.fastq.gz"),
  LongFastQ = NA,
  Fast5 = NA,
  GenomeSize = NA
)

write_csv(
  tb,
  here("config", "nf_runs", glue("ast_browser_samplesheet_{Sys.Date()}.csv"))
)
