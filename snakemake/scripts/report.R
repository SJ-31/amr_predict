library(tidyverse)
library(glue)
library(htmltools)
library(gt)


## * Helper functions

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

## ** Evaluation tables

make_base_table <- function(tb) {
  # Expects that tb has been widened with names_from dataset
  gt(
    tb,
    rowname_col = "metric",
    groupname_col = "task_type",
    row_group_as_column = TRUE
  ) |>
    tab_stubhead(label = "Metric") |>
    fmt_number(decimals = 2)
}

split_tab_by_model <- function(tb, datasets, with_bar = TRUE) {
  lapply(unique(tb$model), \(m) {
    cur <- filter(tb, model == m) |>
      select(-model) |>
      make_base_table() |>
      tab_header(m)
    if (with_bar) {
      # TODO: would like to have an automated coloring
      cur <- cols_nanoplot(
        cur,
        columns = datasets,
        plot_type = "bar",
        new_col_name = glue("{m}_bar"),
        new_col_label = "",
        autohide = FALSE
      ) |>
        cols_move_to_end(glue("{m}_bar"))
    }
    cur
  })
}


write_holdout_tables <- function(tb, outdir, datasets) {
  dir.create(outdir)
  holdout <- filter(tb, evaluation_method == "holdout")
  cols_keep <- c("metric", "task_type", "task")
  fmt <- tb |>
    select(-c(evaluation_method, iter)) |>
    group_by(task, metric, task_type, dataset, model, test_set) |>
    summarise(across(where(is.numeric), mean)) |>
    ungroup() |>
    pivot_wider(names_from = dataset) %>%
    select(sort(names(.))) |>
    # Average across iterations
    relocate(all_of(cols_keep), .before = everything())
  for (test_set_name in unique(holdout$test_set)) {
    cur <- fmt |>
      filter(test_set == test_set_name) |>
      select(-test_set)
    tab <- gt_group(
      .list = split_tab_by_model(cur, datasets = datasets, with_bar = TRUE)
    )
    gtsave(tab, glue("{outdir}/holdout_{test_set_name}.html"))
  }
}

write_cv_tables <- function(tb, outdir, ctrl = FALSE, datasets) {
  dir.create(outdir)
  if (ctrl) fname <- "ctrl_cv" else fname <- "cv"
  cv <- filter(tb, evaluation_method == fname)
  models <- unique(cv$model)
  fmt <- cv |>
    select(-evaluation_method, -iter) |>
    group_by(task, task_type, metric, dataset, model) |>
    summarise(
      mean = mean(value),
      std = sd(value),
      min = min(value),
      max = max(value)
    ) |>
    ungroup() |>
    pivot_longer(cols = c(mean, std, min, max)) |>
    pivot_wider(names_from = dataset)
  for (measure in unique(fmt$name)) {
    cur <- filter(fmt, name == measure) |> select(-name)
    tab <- gt_group(
      .list = split_tab_by_model(cur, datasets = datasets, with_bar = TRUE)
    )
    gtsave(tab, glue("{outdir}/{fname}_{measure}.html"))
  }
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

sample_metadata_table <- function() {
  # TODO: unfinished, do the same for the sample metadata. This
  # will save you having to store a bunch of intermediate output for no reason
}

evaluation_tables <- function() {
  outdir <- snakemake@params[["outdir"]]
  tb <- read_csv(snakemake@input[[1]]) |>
    mutate(task = str_remove(task, "_class$"))
  datasets <- unique(tb$dataset)
  dir.create(outdir, showWarnings = FALSE)
  . <- lmap(snakemake@output, \(x) {
    eval_task <- names(x)
    eval_out <- x[[1]]
    if (eval_task == "holdout") {
      write_holdout_tables(tb, eval_out, datasets = datasets)
    } else if (eval_task == "ctrl_cv") {
      write_cv_tables(tb, eval_out, ctrl = TRUE, datasets)
    } else if (eval_task == "cv") {
      write_cv_tables(tb, eval_out, ctrl = FALSE, datasets)
    }
    list()
  })
}

## * Snakemake entry point

if (exists("snakemake")) {
  globalenv()[[snakemake@rule]]()
}
