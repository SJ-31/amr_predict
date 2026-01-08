library(here)
library(tidyverse)
library(glue)

to_assemble <- read_lines(here("data", "temp", "ast_to_assemble.txt"))
metadata <- read_tsv(here("data", "meta", "ast_subsampled.tsv")) |>
  filter(Run %in% to_assemble)
ast_dir <- here("data", "remote", "ast_browser")

date <- Sys.Date()
meta_all <- read_tsv(here("data", "meta", "ast_subsampled.tsv"))

genome2bsample <- read_csv(here("data", "meta", "genome2biosample.csv"))

## * Bacass samplesheet

data_dir <- "/data/home/shannc/amr_predict/data/remote/ast_browser/raw/"
directories <- list.dirs(ast_dir, recursive = FALSE, full.names = FALSE) |>
  keep(\(x) x %in% metadata$Run)

missing <- directories |>
  discard(
    \(x) {
      file.exists(here(ast_dir, "raw", glue("{x}_1.fastq.gz"))) &&
        file.exists(here(ast_dir, "raw", glue("{x}_2.fastq.gz")))
    }
  )

directories <- directories |> discard(\(x) x %in% missing)

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


# Command for renaming the genome files with brename
# brename -p "(GC._[0-9]+\.[0-9])_.*genomic.fna" -r "\$1.fasta"

## * Funscan samplesheet
#
# %%

funscan_dir <- here("config", "nf_runs", "funcscan")

fsheet <- local({
  path <- "/data/home/shannc/amr_predict/data/remote/genomes/ast_browser_downloaded/"
  download_dir <- here("data", "remote", "genomes", "ast_browser_downloaded")
  from_downloaded <- tibble(
    fasta = list.files(download_dir, pattern = ".fasta")
  ) |>
    mutate(Genome = str_remove(fasta, ".fasta$")) |>
    inner_join(genome2bsample, by = join_by(Genome)) |>
    relocate(BioSample, .before = everything()) |>
    select(-Genome) |>
    rename(sample = "BioSample") |>
    mutate(fasta = paste0(path, fasta))
  ## from_bacass <- tibble(sample = "", fasta = "") # TODO: need to finish these
  from_downloaded
})

chunks <- split(fsheet, ceiling(seq_len(nrow(fsheet)) / 1000))
for (chunk in names(chunks)) {
  fname <- here(funscan_dir, glue("ast_browser_d-{chunk}_{date}.csv"))
  write_csv(chunks[[chunk]], fname)
}
