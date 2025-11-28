suppressMessages({
  library(tidyverse)
  library(here)
  library(glue)
  library(logger)
  library(taxizedb)
  source(here("src", "R", "utils.R"))
})

META <- here("data", "meta")

tdb_cache$cache_path_set(
  full_path = here("data", "remote", "cache", "taxizedb")
)

log_path <- here(META, "biosample_mapping.log")
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
srr_map_file <- here(META, "biosample_mapping_2025-11-21.csv")

standard <- "CLSI"
paths <- list(
  samples = here(META, "ncbi_all_samples.tsv"),
  null_description = here(
    "data",
    "meta",
    "ncbi_all_antibiotic_null_counts.tsv"
  ),
  taxon_counts = here(META, "ncbi_all_taxon_counts.tsv")
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
  lst
}

bioprojects <- read_existing(
  here(META, "ast_bioproject_titles.tsv"),
  \(f) {
    x <- get_bioproject_titles(unique(project_meta$BioProject))
    write_tsv(x, f)
    x
  },
  read_tsv
)

formatted <- read_existing(paths, format_main, read_fn = read_tsv)

project_meta <- read_tsv(ast_file) |>
  distinct(`#BioSample`, .keep_all = TRUE) |>
  filter(`#BioSample` %in% formatted$samples$BioSample) |>
  rename_with(\(x) str_replace_all(str_to_lower(x), " ", "_")) |>
  rename(BioSample = "#biosample") |>
  inner_join(
    read_csv(srr_map_file),
    by = join_by(BioSample)
  ) |>
  inner_join(bioprojects, by = join_by(x$BioProject == y$acc)) |>
  mutate(
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

source_pat <- yaml::read_yaml(here("config", "amr_isolation_recoding.yaml"))

re_paste <- function(vec) {
  regex(paste(vec, collapse = "|"), TRUE)
}

project_meta$isolation_source_broad <- with(
  project_meta,
  case_when(
    str_detect(isolation_source, re_paste(source_pat$animal)) ~ "animal",
    str_detect(isolation_source, re_paste(source_pat$gi)) ~ "gi_fecal",
    str_detect(isolation_source, re_paste(source_pat$resp)) ~ "respiratory",
    str_detect(isolation_source, re_paste(source_pat$eye)) ~ "eye",
    str_detect(isolation_source, re_paste(source_pat$skin)) ~
      "skin/soft_tissue",
    str_detect(isolation_source, re_paste(source_pat$accessory)) ~
      "skin_accessory",
    str_detect(isolation_source, re_paste(source_pat$wound)) ~ "wound",
    str_detect(isolation_source, re_paste(source_pat$fluid)) ~
      "sterile_body_fluid",
    str_detect(isolation_source, re_paste(source_pat$tissue)) ~ "tissue/organ",
    str_detect(isolation_source, re_paste(source_pat$urogen)) ~ "urogenital",
    str_detect(isolation_source, re_paste(source_pat$surgical)) ~
      "surgical/implant",
    str_detect(isolation_source, re_paste(source_pat$joint)) ~
      "joint/orthopedic",
    str_detect(isolation_source, re_paste(source_pat$food_animal)) ~
      "animal_product",
    str_detect(isolation_source, re_paste(source_pat$food_plant)) ~
      "plant_product",
    str_detect(isolation_source, re_paste(source_pat$env)) ~ "environmental",
    str_detect(isolation_source, re_paste(source_pat$unknown)) ~ "unknown",
    is.na(isolation_source) ~ NA,
    .default = "unknown"
  )
)


bsample_attributes <- read_existing(
  here(META, "biosample_attributes.tsv"),
  \(f) {
    chunk_size <- 2000
    v <- project_meta$BioSample
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
  select(project_meta, c(BioSample, title)),
  by = join_by(x$acc == y$BioSample)
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

# Recoding rules for umbrella projects
# - Unify all NARMS projects as one because AST is centralized
# - Rename Vet-LIRN projects to enable grouping by site of collection and sequencing rather than by species
# - Leave GenomeTrakr as is because they provide no detailed data on collection

bsample_attributes$ast_by <- with(
  bsample_attributes,
  case_when(
    str_detect(project_name, "GenomeTrakr.*NARMS") ~ "NARMS",
    str_detect(title, "NARMS") ~ "NARMS",
    .default = collected_by
  )
)
bsample_attributes$umbrella_project <- with(
  bsample_attributes,
  case_when(
    str_detect(project_name, "GenomeTrakr.*NARMS") ~ "NARMS",
    str_detect(title, "NARMS") ~ "NARMS",
    str_detect(title, "Vet-LIRN") ~ "Vet-LIRN",
    str_detect(title, "GenomeTrakr") ~ "GenomeTrakr",
    .default = NA
  )
)

## ** Re-join with all metadata

cols_remove <- c(
  "AssemblyName",
  "Query",
  "ReleaseDate",
  "SampleType",
  "ScientificName",
  "antibiotic",
  "create_date",
  "disk_diffusion_(mm)",
  "isolate",
  "isolation_type",
  "mic_(mg/l)",
  "resistance_phenotype",
  "testing_standard",
  "organism",
  "organism_group",
  "scientific_name",
  "source",
  "vendor"
)
wanted_attributes <- c(
  "collection_year",
  "sequenced_by",
  "umbrella_project",
  "serovar",
  "serotype",
  "strain"
)
project_meta <- inner_join(
  project_meta,
  select(bsample_attributes, c(acc, all_of(wanted_attributes))),
  by = join_by(x$BioSample == y$acc)
) |>
  mutate(
    umbrella_project = case_match(
      BioProject,
      "PRJNA600010" ~ "GenomeTrakr_Canada",
      "PRJNA966974" ~ "GenomeTrakr_Canada",
      "PRJNA1148950" ~ "GenomeTrakr_Canada",
      "PRJNA454819" ~ "GenomeTrakr_Canada",
      "PRJNA435747" ~ "GenomeTrakr_Canada",
      "PRJNA417863" ~ "GenomeTrakr_Canada",
      .default = umbrella_project
    )
  ) |>
  select(-all_of(cols_remove))


## * Antibiotic categorization

## ** Add drug classes

am_custom <- yaml::read_yaml(here("config", "antimicrobials_custom.yaml"))

am_cat <- read_csv(here(META, "ADB_all_compounds.csv")) |>
  rename_with(\(x) str_replace_all(str_to_lower(x), " ", "_")) |>
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
    tax_lookup <- setNames(project_meta[[unit]], project_meta$BioSample)
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
    filter(BioSample %in% inter) |>
    select(BioSample, genus, order, species)
})
intersections <- discard(intersections, \(x) nrow(x) == 0)
# NOTE: there is potentially overlap between the WHO groups
# because it's possible for a given pathogen to meet the criteria of multiple groups
# based on their resistance classes.
# EX: A Shigella sample that is resistant to a fluoroquinolone drug,
# but is also resistant to a third-gen beta-lactam would meet the criteria for WHO
# critical and WHO high

project_meta$interest_group <- with(
  project_meta,
  case_when(
    BioSample %in% amr_groups$WHO_critical ~ "WHO_critical",
    BioSample %in% amr_groups$WHO_high ~ "WHO_high",
    BioSample %in% amr_groups$WHO_medium ~ "WHO_medium",
    .default = NA
  )
)


#' Convert a list of names->vector_of_ids (where ids may be duplicated)
#' into a mapping vector names->ids
#'
#' @param factor_list A list where names are levels of a factor,
#' and the values are identifiers
#' e.g. for a factor with levels A,B,C:
#' list(A = c("i1", "i42", "i97"), B = c("i12", "i554"), C = "i07")
#' @param unique_fn Function to reconcile ids assigned to multiple levels
#' @return
#'  A vector mapping ids to their levels
levels2tb <- function(
  factor_list,
  unique_fn = first
) {
  tb <- lmap(factor_list, \(n) {
    as_tibble_col(n[[1]], column_name = names(n)) |> pivot_longer(names(n))
  }) |>
    bind_rows() |>
    group_by(value) |>
    summarise(name = unique_fn(name))
  with(tb, setNames(name, value))
}

project_meta <- mutate(
  project_meta,
  interest_group = levels2tb(amr_groups)[project_meta$BioSample]
) |>
  mutate(interest_group = replace_na(interest_group, "none"))

# %%

## * Subsampling

# REVIEW: you could also weight by genera

final_selection <- local({
  tbs <- list()

  project_meta <- distinct(project_meta, BioSample, .keep_all = TRUE)

  # Subsample evenly from the big 4 projects.
  tbs$proj_group <- project_meta |>
    filter(!is.na(umbrella_project)) |>
    group_by(umbrella_project, genus) |>
    slice_sample(prop = 0.2)

  tax_counts <- table(project_meta$species)
  tax_quantiles <- quantile(tax_counts, c(0.6, 0.9)) # Counts are highly skewed
  tax_representation <- case_when(
    tax_counts >= tax_quantiles["90%"] ~ "majority",
    tax_counts < tax_quantiles["60%"] ~ "minority",
    .default = "intermediate",
  ) |>
    `names<-`(names(tax_counts))
  project_meta$species_representation <- tax_representation[
    project_meta$species
  ]

  # If genus has less than 100 samples, will take all of them
  tbs$minority_genera <- project_meta |>
    group_by(genus) |>
    filter(n() <= 100)

  # Add samples from groups of interest
  tbs$interest <- project_meta |>
    group_by(interest_group) |>
    slice_sample(n = 500, replace = TRUE)

  kept_ids <- lapply(tbs, \(x) x$BioSample) |>
    unlist() |>
    unique()

  project_meta |>
    filter(BioSample %in% kept_ids) |>
    relocate(
      species,
      subspecies,
      genus,
      family,
      order,
      .after = "BioSample"
    )
})
write_tsv(final_selection, here(META, "ast_subsampled.tsv"))
write_lines(
  final_selection$Run,
  here(META, "ast_subsampled_runs.txt")
)

confounding_score_multi(
  final_selection,
  c("interest_group", "genus", "umbrella_project", "collection_year")
)
