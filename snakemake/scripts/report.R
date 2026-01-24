library(tidyverse)
library(glue)
library(gt)


## * Helper functions

add_model_spanner <- function(tab, cur_model, last_dset) {
  pref <- glue("{cur_model}_")
  tab <- tab_spanner(
    tab,
    label = cur_model,
    columns = starts_with(pref),
    level = 2
  ) |>
    cols_label_with(
      columns = starts_with(pref),
      fn = \(x) str_remove(x, pref)
    )
  # TODO: would like to have an automated coloring
  tab <- tab |>
    cols_nanoplot(
      starts_with(pref),
      plot_type = "bar",
      new_col_name = glue("{pref}_bar"),
      new_col_label = "",
## ** Metadata tables

make_group_count_tables <- function(tb) {
  tables <- apply(tb, 1, \(row) {
    tmp <- read_csv(row["filenames"])
    gt(tmp) |>
      fmt_number(columns = "Proportion", decimals = 2) |>
      opt_interactive(
        use_search = TRUE,
        use_highlight = TRUE,
        use_compact_mode = TRUE
      ) |>
      tab_header(row["group"])
    # TODO: can prettify this
  })
  tables
}

#' Save a list of gt_tbl objects as a single file
#'
#' @description
#' [2026-01-24 Sat] This is primarily a workaround for gtsave not working on
#' gt_group objects with interactive components
save_gt_list <- function(tabs, outdir) {
  got_lib <- FALSE
  dir.create(outdir)
  lib <- glue("{outdir}/lib")
  html_parts <- lapply(tabs, \(t) {
    dir <- tempdir()
    temp_file <- tempfile(fileext = ".html", tmpdir = dir)
    gtsave(t, filename = temp_file)
    cur_lib <- glue("{dir}/lib")
    if (!got_lib && dir.exists(cur_lib)) {
      dir.create(lib)
      file.copy(cur_lib, outdir, recursive = TRUE)
      got_lib <<- TRUE
    }
    readLines(temp_file)
  })
  file <- glue("{outdir}/index.html")
  tryCatch(
    {
      combined_html <- unlist(html_parts)
      writeLines(combined_html, file)
    },
    error = \(e) {
      unlink(outdir, recursive = TRUE)
      stop(e)
    }
  )
}

#' Generate count tables with gt from the csvs written by `write_seq_meta_tables`
#' in report.py
#'
write_seq_anno_counts <- function(table_csvs, outdir, prefix) {
  table_csvs <- table_csvs |> keep(\(x) str_starts(glue("{prefix}_"), x))
  tb <- tibble(filenames = list.files(table_csvs, full.names = TRUE)) |>
    mutate(
      tmp = basename(filenames),
      tool = map_chr(tmp, \(x) str_split_1(x, "_")[1]),
      group = map_chr(tmp, \(x) str_remove(x, ".*?_") |> str_remove(".csv$"))
    ) |>
    select(-tmp)
  tabs <- make_group_count_tables(tb)
  save_gt_list(tabs, outdir)
}

}


## ** Holdout tables

make_holdout_table <- function(tb) {
  models <- unique(tb$model)
  cols_keep <- c("metric", "task_type")
  formatted <- tb |>
    select(-c(evaluation_method, test_set, iter)) |>
    group_by(task, metric) |>
    summarise(across(where(is.numeric), mean)) |>
    # Average across iterations
    ungroup() |>
    select(-task) |>
    pivot_wider(names_from = c(model, dataset)) %>%
    select(sort(names(.))) |>
    relocate(all_of(cols_keep), .before = everything())
  last_dset <- colnames(formatted) |>
    tail(n = 1) |>
    str_remove(".*?_")
  tab <- gt(
    formatted,
    rowname_col = "metric",
    groupname_col = "task_type",
    row_group_as_column = TRUE
  ) |>
    tab_stubhead(label = "Metric") |>
    fmt_number(decimals = 2)
  for (m in models) {
    tab <- add_model_spanner(tab, cur_model = m, last_dset = last_dset)
  }
  tab |>
    tab_spanner(label = md("**Model**"), spanners = models)
}

write_holdout_tables <- function(tb, outdir) {
  holdout <- filter(tb, evaluation_method == "holdout") |>
    mutate(task = str_remove(task, "_class$"))
  for (test_set_name in unique(holdout$test_set)) {
    cur <- holdout |> filter(test_set == test_set_name)
    tab <- make_holdout_table(cur)
    cur |> gtsave(glue("{outdir}/holdout_{test_set_name}.html"))
  }
  # TODO: Write one table for each test set.
  # In snakemake, subcategory will be "Holdout (table)" and label by test set name
}

## * Rules

seq_metadata_tables <- function() {
  csv_path <- snakemake@input[[1]]
  prefix <- snakemake@params[["tool_prefix"]]
  tb <- tibble(filenames = list.files(csv_path, full.names = TRUE)) |>
    mutate(
      tmp = basename(filenames),
      tool = map_chr(tmp, \(x) str_split_1(x, "_")[1]),
      group = map_chr(tmp, \(x) str_remove(x, ".*?_") |> str_remove(".csv$"))
    ) |>
    select(-tmp) |>
    filter(tool == prefix)
  tabs <- make_group_count_tables(tb)
  save_gt_list(tabs, snakemake@output[[1]])
}
# TODO: unfinished
evaluation_tables <- function() {
  tb <- read_csv(snakemake@input[1])
  write_holdout_tables(tb)
}

## * Snakemake entry point

if (exists("snakemake")) {
  globalenv()[[snakemake@rule]]()
}
