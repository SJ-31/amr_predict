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
      autohide = FALSE
    ) |>
    cols_move(glue("{pref}_bar"), after = glue("{cur_model}_{last_dset}"))
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

# TODO: unfinished
evaluation_tables <- function() {
  tb <- read_csv(snakemake@input[1])
  write_holdout_tables(tb)
}

## * Snakemake entry point

if (exists("snakemake")) {
  globalenv()[[snakemake@rule]]()
}
