suppressMessages({
  library(tidyverse)
  library(here)
  library(glue)
  library(BiocFileCache)
  library(rentrez)
})


#' Map biosample accession `acc` to any available databases
#'
biosample_db_links <- function(acc, cache) {
  result <- tibble(db = "BioSample", value = acc)
  search_biosample <- lst <- xml <- NULL

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
        xml <- entrez_fetch(
          db = "biosample",
          id = search_biosample,
          rettype = "xml",
          retmode = "text"
        )
      },
      silent = TRUE
    )
  }

  if (!is.null(xml)) {
    lst <- xml2::as_list(xml2::as_xml_document(xml))
    savepath <- bfcnew(cache, acc, ext = ".rds")
    saveRDS(lst, file = savepath)
  }

  if (!is.null(lst)) {
    bsample_meta <- lapply(
      lst$BioSampleSet$BioSample$Ids,
      \(x) {
        tibble_row(db = attr(x, "db"), value = x[[1]])
      }
    ) |>
      bind_rows()
    result <- bind_rows(result, bsample_meta)
  }

  distinct(result) |> pivot_wider(names_from = db)
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
  args <- parse_args(parser)
  accs <- read_lines(args$input)
  cache <- BiocFileCache(args$cache)
  lapply(accs, \(x) biosample_db_links(x, cache = cache)) |>
    bind_rows() |>
    write_tsv(args$output)
}
