library(tidyverse)
library(here)
# Generate mapping files for the bacass-assembled AST browser genomes
# for use with `genome_qc.py`
# Three files to generate, written into the `temp` directory:
# - kraken2_expected_taxids
# - query2taxid
# - taxid2reference

remote <- here("data", "remote")
meta <- here("data", "meta")
temp <- here("data", "temp")

from_assembled <- tibble(
  query = list.files(here(
    remote,
    "output/ast_browser/bacass_2025-12-05/Kraken2"
  ))
) |>
  mutate(query = str_remove(query, ".kraken2.report.txt"))
metadata <- read_tsv(here(meta, "ast_subsampled.tsv"))
merged <- from_assembled |>
  inner_join(metadata, by = join_by(x$query == y$BioSample))

# kraken2_expected_taxids file
merged |>
  select(query, TaxID) |>
  write_tsv(here(temp, "ast_subsampled_sample2taxid.tsv"), col_names = FALSE)

ref <- read_tsv(here(meta, "reference_genomes.tsv"))
ref_dir <- here(remote, "genomes/reference/ncbi_dataset/data")
genome_dir <- here(remote, "genomes/ast_browser_batches/6_bacass")
available_refs <- tibble(dir = list.files(ref_dir, full.names = TRUE)) |>
  mutate(ref = basename(dir)) |>
  filter(dir.exists(dir)) |>
  mutate(file = map_chr(dir, \(x) head(list.files(x), 1))) |>
  select(-dir)
ref <- ref |>
  inner_join(available_refs, by = join_by(x$`Assembly Accession` == y$ref))

query2taxid <- tibble(file = list.files(genome_dir, full.names = TRUE)) |>
  mutate(BioSample = str_remove(basename(file), ".fasta$")) |>
  inner_join(metadata, by = join_by(BioSample)) |>
  select(file, TaxID)

taxid2ref <- ref |> select(taxid_query, file)

taxid2ref |>
  write_tsv(here(temp, "ast_subsampled_taxid2ref.tsv"), col_names = FALSE)
query2taxid |>
  write_tsv(here(temp, "ast_subsampled_query2taxid.tsv"), col_names = FALSE)
