suppressMessages({
  library(tidyverse)
  library(here)
  library(glue)
  library(logger)
  library(taxizedb)
  source(here("src", "R", "utils.R"))
})

tdb_cache$cache_path_set(
  full_path = here("data", "remote", "cache", "taxizedb")
)

log_path <- here("data", "meta", "biosample_mapping.log")
if (file.exists(log_path)) {
  file.remove(log_path)
}
lf <- appender_file(log_path)

set.seed(3110)

log_layout(layout = layout_blank, namespace = "tb_view")
log_formatter(formatter = formatter_pander, namespace = "tb_view")
log_appender(lf)
log_appender(lf, namespace = "tb_view")

ast_file <- here("data", "raw", "asts.tsv")
srr_map_file <- here("data", "meta", "biosample_mapping_2025-11-21.csv")

standard <- "CLSI"
paths <- list(
  samples = here("data", "meta", "ncbi_all_samples.tsv"),
  null_description = here(
    "data",
    "meta",
    "ncbi_all_antibiotic_null_counts.tsv"
  ),
  taxon_counts = here("data", "meta", "ncbi_all_taxon_counts.tsv")
)

format_main <- function(lst) {
  bsample_map <- read_csv(srr_map_file) |>
    filter(!is.na(BioSample) & !is.na(Run))

  ast <- read_tsv(ast_file)

  joined <- inner_join(
    bsample_map,
    ast,
    by = join_by(x$BioSample == y$`#BioSample`)
  ) |>
    filter(!is.na(`MIC (mg/L)`) & `Testing standard` == standard) |>
    rename(MIC = "MIC (mg/L)") |> # MIC measured in (mg/L)
    mutate(
      Antibiotic = case_match(
        Antibiotic,
        "cefalotin" ~ "cephalotin",
        .default = Antibiotic
      )
    )
  # TODO: how to handle "Measurement sign"?

  # Unique samples
  utb <- joined |> distinct(BioSample, .keep_all = TRUE)

  # Number of samples with data
  n_samples <- length(utb$BioSample)
  log_info("Number of samples with data: {n_samples}")

  mic_values <- joined |>
    select(BioSample, Run, ScientificName, MIC, Antibiotic) |>
    pivot_wider(
      names_from = Antibiotic,
      values_from = MIC,
      id_cols = c(BioSample, Run, ScientificName),
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
  ) |>
    distinct(BioSample, .keep_all = TRUE)

  null_description <- colSums(is.na(select(
    mic_values,
    -c(BioSample, Run, ScientificName)
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
    write_tsv(lst$taxon_counts)
  write_tsv(final, lst$samples)
  write_tsv(null_description, lst$null_description)
}

bioprojects <- read_existing(
  here("data", "meta", "ast_bioproject_titles.tsv"),
  \(f) {
    x <- get_bioproject_titles(unique(project_meta$BioProject))
    write_tsv(x, f)
    x
  },
  read_tsv
)

formatted <- read_existing(paths, format_main, read_fn = read_tsv)

# TODO: organism group might easier to work with than species
# TODO: figure out a sampling procedure for project

project_meta <- read_tsv(ast_file) |>
  distinct(`#BioSample`, .keep_all = TRUE) |>
  filter(`#BioSample` %in% formatted$samples$BioSample) |>
  rename_with(\(x) str_to_lower(x) |> str_replace_all(" ", "_")) |>
  inner_join(
    read_csv(srr_map_file),
    by = join_by(x$`#biosample` == y$BioSample)
  ) |>
  inner_join(bioprojects, by = join_by(x$BioProject == y$acc)) |>
  rename(date = "ReleaseDate") |>
  mutate(
    year = year(date),
    isolation_source = str_to_lower(isolation_source) |>
      str_replace_all("[ -]", "_"),
    species = ncbi_taxid2rank(TaxID, "species")[
      TaxID
    ],
    subspecies = ncbi_taxid2rank(TaxID, "subspecies")[
      TaxID
    ],
    genus = ncbi_taxid2rank(TaxID, "genus")[
      TaxID
    ],
    order = ncbi_taxid2rank(TaxID, "order")[
      TaxID
    ],
    family = ncbi_taxid2rank(TaxID, "family")[
      TaxID
    ]
  )


bproject_counts <- as.data.frame(table(project_meta$BioProject)) |>
  as_tibble() |>
  arrange(desc(Freq)) |>
  rename(acc = "Var1") |>
  inner_join(bioprojects, by = join_by(acc))

# Upper limit for range in mg/L concentrations is 1024

## * Recoding

# Majority of samples from NARMS
# CVM: Center for Veterinary Medicine
# FSIS: Food Safety and Inspection Service

# TODO: will also need to filter samples by QC metrics after you've downloaded and assembled them.
# Mainly check assembly metrics cause the submitters will hopefully have handled raw reads

# TODO: when downloading, first check if a fasta assembly is available on the ncbi

# TODO: recode the isolation source, making it consistent across projects
# or could just change all the NARMS as "retail_meat"

# Recoding rules for umbrella projects
# - Unify all NARMS projects as one because AST is centralized
# - Rename Vet-LIRN projects to enable grouping by site of collection and sequencing rather than by species
# - Leave GenomeTrakr as is because they provide no detailed data on collection

# TODO: Get collection date...

bsample_attributes <- read_existing(
  here("data", "meta", "biosample_attributes.tsv"),
  \(f) {
    chunk_size <- 2000
    v <- project_meta$`#biosample`
    splits <- split(v, ceiling(seq_along(v) / chunk_size))
    attr <- lapply(splits, \(chunk) {
      get_biosample_attrs(chunk)
    }) |>
      bind_rows()
    write_tsv(attr, f)
    attr
  },
  read_tsv
)
repeated <- c("project_name", "isolate", "isolation_source")
for (r in repeated) {
  bsample_attributes <- unite(
    bsample_attributes,
    !!as.symbol(r),
    contains(glue("{r}.")),
    na.rm = TRUE,
    sep = "_",
    remove = TRUE
  )
}
bsample_attributes <- inner_join(
  bsample_attributes,
  select(project_meta, c(`#biosample`, title)),
  by = join_by(x$acc == y$`#biosample`)
)

## ** Add collection dates

bsample_attributes$collection_year <- lapply(
  str_remove_all(bsample_attributes$collection_date, "[A-Za-z ]+"),
  \(x) {
    count <- str_count(x, "-")
    if (is.na(count)) {
      NA
    } else if (count == 1) {
      year(ym(x))
    } else if (count == 2) {
      year(ymd(x))
    } else {
      year(suppressWarnings(parse_date(x, "%Y")))
    }
  }
) |>
  unlist()

## ** Unify sample handling

vet_lirn <- filter(bsample_attributes, str_detect(title, "Vet-LIRN"))
gtrakr_narms <- bsample_attributes |>
  filter(str_detect(project_name, "GenomeTrakr; NARMS")) |>
  pluck("acc")

## TODO: unfinished [2025-11-27 Thu]
## project_meta$sample_handling <- case_when(
##   project_meta$`#biosample` %in% vet_lirn$acc ~
##     paste0("Vet-LIRN", vet_lirn$sequenced_by, vet_lirn$collected_by),
##   str_detect(project_meta$title, "NARMS") ~
##     str_extract(project_meta$title, "(.* NARMS) .*", group = 1),
##   project_meta$`#biosample` %in% gtrakr_narms ~ "CVM NARMS",
##   .default = project_meta$title
## )

## ** Re-join with all metadata

wanted_attributes <- c("collection_date")
project_meta <- inner_join(
  project_meta,
  select(bsample_attributes, c(acc, all_of(wanted_attributes))),
  by = join_by(x$`#biosample` == y$acc)
)

## * Antibiotic categorization

## ** Add drug classes

am_custom <- yaml::read_yaml(here("config", "antimicrobials_custom.yaml"))

am_cat <- read_csv(here("data", "meta", "ADB_all_compounds.csv")) |>
  rename_with(\(x) str_to_lower(x) |> str_replace_all(" ", "_")) |>
  mutate(
    drug_name = str_to_lower(
      case_match(
        drug_name,
        "Daptomycin (Cubicin)" ~ "daptomycin",
        "Quinupristin/dalfopristin (Synercid)" ~ "synercid",
        "Amoxicillin + clavulanic acid" ~ "amoxicillin-clavulanic acid",
        "Sulfamethoxazole+trimethoprim" ~ "trimethoprim-sulfamethoxazole",
        "Cephalotin" ~ "cephalothin",
        "Rifampicin" ~ "rifampin",
        "Tedizolid (Sivextro)" ~ "tedizolid",
        "Penicillin V" ~ "penicillin",
        "Penicillin G" ~ "benzylpenicillin",
        "Avycaz/Zavicefta (Ceftazidime/Avibactam)" ~ "ceftazidime-avibactam",
        "Metronidazole (Flagyl)" ~ "metronidazole",
        "Ceftolozane + tazobactam (Zerbaxa)" ~ "ceftolozane-tazobactam",
        "Meropenem + Vaborbactam" ~ "meropenem-vaborbactam",
        "Cefiderocol (Fetroja)" ~ "cefiderocol",
        "Imipenem + cilastatin/relebactam" ~ "imipenem-relebactam",
        "Flucytosine (prodrug)" ~ "flucytosine",
        "Plazomicin (Zemdri)" ~ "plazomicin",
        "Oxytetracycline (terramycin)" ~ "oxytetracycline",
        "Ceftizoxime alapivoxil (prodrug)" ~ "ceftizoxime",
        .default = drug_name
      )
    ),
    drug_class = str_trim(str_remove(
      str_to_lower(drug_class),
      "drug combination: "
    ))
  )

present_am <- local({
  d <- colnames(formatted$samples) |>
    discard(\(x) {
      str_ends(x, "_class") | x %in% c("BioSample", "Run", "ScientificName")
    }) |>
    str_to_lower()
  left_join(tibble(drug_name = d), am_cat, by = join_by(drug_name)) |>
    mutate(
      drug_class = case_when(
        is.na(drug_class) ~ unlist(am_custom$AM_classes)[drug_name],
        .default = drug_class
      )
    )
})

## ** Label AM groups
# %%

dclass2names <- group_by(present_am, drug_class) |>
  summarise(drug_name = list(drug_name)) |>
  filter(!is.na(drug_class))


drug_class_lookups <- sapply(
  am_custom$AM_of_interest,
  \(x) {
    filter(dclass2names, str_detect(drug_class, x)) |>
      pluck("drug_name") |>
      unlist()
  },
  USE.NAMES = TRUE,
  simplify = FALSE
)

samples <- formatted$samples$BioSample
amr_groups <- list()
group_tracker <- list()
for (group in names(am_custom$AM_groups)) {
  org_groups <- am_custom$AM_groups[[group]]
  mask <- lapply(seq_along(org_groups), \(i) {
    spec <- org_groups[[i]]
    tmatch <- spec$exact
    unit <- names(tmatch)
    tax_lookup <- setNames(project_meta[[unit]], project_meta$`#biosample`)
    taxon_match <- tax_lookup[samples] %in% tmatch[[unit]]
    if (!is.null(spec$drug_name)) {
      relevant_drugs <- spec$drug_name
    } else {
      relevant_drugs <- lapply(
        spec$drug_class_re,
        \(x) drug_class_lookups[[x]]
      ) |>
        unlist()
    }
    relevant_drugs <- keep(
      relevant_drugs,
      \(x) x %in% colnames(formatted$samples)
    )
    if (length(relevant_drugs) == 0) {
      return(rep(FALSE, nrow(formatted$samples)))
    }
    relevant_drugs <- paste0(relevant_drugs, "_class")
    m <- formatted$samples |>
      mutate(mask = if_any(relevant_drugs, \(x) x == "resistant")) |>
      pluck("mask")
    group_tracker[[glue("{group}_{i}")]] <<- formatted$samples$BioSample[
      m & taxon_match
    ]
    m & taxon_match
  }) |>
    reduce(\(x, y) x | y)
  amr_groups[[group]] <- formatted$samples$BioSample[mask] |> discard(is.na)
}

# Check overlap
intersections <- list()
tmp <- apply(combn(names(group_tracker), 2), 2, \(col) {
  g1 <- col[1]
  g2 <- col[2]
  inter <- intersect(group_tracker[[g1]], group_tracker[[g2]])
  intersection <- length(inter)
  log_info("intersection between {g1} {g2}: {intersection}")
  intersections[[glue("{g1}_{g2}")]] <<- project_meta |>
    filter(`#biosample` %in% inter) |>
    select(`#biosample`, genus, order, species)
})
intersections <- discard(intersections, \(x) nrow(x) == 0)
# NOTE: there is potentially overlap between the WHO groups
# because it's possible for a given pathogen to meet the criteria of multiple groups
# based on their resistance classes.
# EX: A Shigella sample that is resistant to a fluoroquinolone drug,
# but is also resistant to a third-gen beta-lactam would meet the criteria for WHO
# critical and WHO high

# %%

crit <- project_meta |> filter(`#biosample` %in% amr_groups$WHO_critical)

## * Subsampling

# Samples from GenomeTrakr, which comprises multiple laboratories
gtrak <- bproject_counts |> filter(str_detect(title, "GenomeTrakr"))

# TODO: subsample evenly from the big 3 projects.
kept_ids <- c()

# Add all samples that are species of interest
for (g in amr_groups) {
  kept_ids <- c(kept_ids, sample(g, size = 500))
}
## kept_ids <- c(
##   kept_ids,
##   unique(formatted$samples$BioSample %in% unlist(amr_groups, use.names = FALSE))
## )
