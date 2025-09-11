suppressMessages({
  library(tidyverse)
  library(here)
  library(glue)
  library(rentrez)
})


#' Map biosample accession `acc` to any available databases
#'
biosample_db_links <- function(acc) {
  search_biosample <- entrez_search(db = "biosample", term = acc)

  result <- tibble(db = "BioSample", value = acc)

  if (length(search_biosample$ids) > 0) {
    xml <- entrez_fetch(
      db = "biosample",
      id = search_biosample,
      rettype = "xml",
      retmode = "text"
    )
    lst <- xml2::as_list(xml2::as_xml_document(xml))
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
    c("-o", "--output"),
    type = "character",
    help = "Output tsv file"
  )
  args <- parse_args(parser)
  accs <- read_lines(args$input)
  lapply(accs, biosample_db_links) |>
    bind_rows() |>
    write_tsv(args$output)
}
