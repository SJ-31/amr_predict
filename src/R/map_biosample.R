suppressMessages({
  library(tidyverse)
  library(here)
  library(glue)
  library(BiocFileCache)
  library(rentrez)
})

# Use esearch to map biosample accessions directly to sra runs
get_sra_runinfo <- function(acc, cache) {
  empty <- tibble(Query = acc)
  tried_fetch <- FALSE

  cache_lookup <- bfcquery(cache, acc)
  if (nrow(cache_lookup) > 0) {
    result <- read_csv(cache_lookup$rpath)
  } else {
    try_search <- system2(
      "esearch",
      args = c("-db", "sra", "-query", acc),
      stdout = TRUE
    )
    fetch <- system2(
      "efetch",
      args = c("-format", "runinfo"),
      input = try_search,
      stdout = TRUE
    )
    savepath <- bfcnew(cache, acc, ext = ".csv")
    if (length(fetch) == 0) {
      empty$EsearchLookup <- "FAILED"
      write_csv(empty, file = savepath)
      result <- empty
    } else {
      empty$EsearchLookup <- "SUCCESS"
      header <- str_split_1(fetch[1], ",")
      result <- lapply(
        fetch[2:length(fetch)],
        \(x) {
          as.data.frame(matrix(str_split_1(x, ","), nrow = 1)) |>
            `colnames<-`(header)
        }
      ) |>
        bind_rows() |>
        mutate(Query = acc, EsearchLookup = "SUCCESS")
      write_csv(result, file = savepath)
    }
  }
  result
}


#' Map biosample accession `acc` to any available databases
#'
biosample_db_links <- function(acc, cache) {
  result <- tibble(db = "BioSample", value = acc)
  search_biosample <- lst <- xml <- NULL
  tried_fetch <- api_lookup_success <- FALSE

  cache_lookup <- bfcquery(cache, acc)
  if (nrow(cache_lookup) > 0) {
    lst <- readRDS(cache_lookup$rpath)
  } else {
    try(search_biosample <- entrez_search(db = "biosample", term = acc))
  }

  if (
    is.null(lst) &&
      !is.null(search_biosample) &&
      length(search_biosample$ids) > 0
  ) {
    try(
      {
        tried_fetch <- TRUE
        xml <- entrez_fetch(
          db = "biosample",
          id = search_biosample,
          rettype = "xml",
          retmode = "text"
        )
        api_lookup_success <- TRUE
      },
      silent = TRUE
    )
  }

  if (!is.null(xml)) {
    lst <- xml2::as_list(xml2::as_xml_document(xml))
    savepath <- bfcnew(cache, acc, ext = ".rds")
    saveRDS(lst, file = savepath)
  } else if (tried_fetch) {
    savepath <- bfcnew(cache, acc, ext = ".rds")
    print(glue("Recording acc {acc} as failed..."))
    saveRDS("FAILED", file = savepath)
  }

  if (!is.null(lst) && lst != "FAILED") {
    bsample_meta <- lapply(
      lst$BioSampleSet$BioSample$Ids,
      \(x) {
        tibble_row(db = attr(x, "db"), value = x[[1]])
      }
    ) |>
      bind_rows()
    result <- bind_rows(result, bsample_meta)
  }

  distinct(result) |>
    mutate(api_lookup_success = api_lookup_success) |>
    pivot_wider(names_from = db)
}


if (sys.nframe() == 0) {
  library(optparse)
  parser <- OptionParser()
  parser <- add_option(
    parser,
    c("-i", "--input"),
    type = "character",
    help = "Input accession list"
  )
  parser <- add_option(
    parser,
    c("-c", "--cache"),
    type = "character",
    help = "Path to cache",
    default = ".rentrez"
  )
  parser <- add_option(
    parser,
    c("-o", "--output"),
    type = "character",
    help = "Output tsv file"
  )
  parser <- add_option(
    parser,
    c("-z", "--with_rentrez"),
    type = "logical",
    help = "Whether to map samples using rentrez",
    action = "store_true",
    default = FALSE
  )
  args <- parse_args(parser)
  accs <- read_lines(args$input)
  cache <- BiocFileCache(args$cache)
  if (args$with_rentrez) {
    map_fn <- biosample_db_links
  } else {
    map_fn <- get_sra_runinfo
  }
  map_fn <- slowly(map_fn, rate = rate_delay(1))
  lapply(accs, \(x) map_fn(x, cache = cache)) |>
    lapply(\(x) {
      if ("BioSample" %in% colnames(x)) {
        unnest(x, BioSample)
      } else {
        x
      }
    }) |>
    bind_rows() |>
    write_tsv(args$output)
}
