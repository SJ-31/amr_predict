read_existing <- function(filename, expr, read_fn = identity) {
  if (class(filename) != "list") {
    if (file.exists(filename)) {
      read_fn(filename)
    } else {
      expr(filename)
    }
  } else {
    if (all(map_lgl(filename, file.exists))) {
      sapply(filename, read_fn, simplify = FALSE, USE.NAMES = TRUE)
    } else {
      expr(filename)
    }
  }
}

table2tb <- function(table, row_header) {
  table |>
    as.data.frame() |>
    `colnames<-`(c(row_header, "Var2", "Freq")) |>
    tidyr::pivot_wider(names_from = Var2, values_from = Freq)
}

describe_na <- function(tb, colname = "col") {
  colSums(is.na(tb)) |>
    sort() |>
    as.data.frame() |>
    `colnames<-`("na_count") |>
    rownames_to_column(var = colname) |>
    mutate(na_percent = na_count / nrow(tb))
}

#' Estimate the degree of confounding in a tabular object `obj`
#' @param x a factor/character column of `obj`
#' @param y a factor/character column of `obj`
#'
#'
#' @description
#'
#' The column confonding score cf is defined for a factor x and some level of y_i of factor y
#' Define v to be the vector of counts of levels of x where x == y_i
#' cf(x, y_i) = sum(v - (1 / length(v))) / mv
#' Where mv is the maximum value of this operation
#'
#' The score ranges from 0-1, where 0 is no confounding and all values of
#' x are evenly distributed. The max score of 1 occurs when all but one levels of x nonzero are in one level of y
#'
#' @return
#' avg: the average of the score of the x vs y and y vs x comparison, which makes the function symmetric
#' y_cols: the result
#'  A score of 1 in a level y_i indicates that all samples labeled y_i are labeled with some specific level of x
confounding_score <- function(obj, x, y) {
  helper <- function(i_x, i_y) {
    tab <- table(
      obj[[i_x]],
      obj[[i_y]]
    ) |>
      as.matrix()
    # Convert contingency table into table of proportions
    tab_prop <- proportions(tab, margin = 2)
    expected_prop <- 1 / nrow(tab)
    deviation <- abs(t(t(tab_prop) - expected_prop))
    max_val <- (expected_prop * (nrow(tab) - 1)) + (1 - expected_prop)
    dev_sum <- colSums(deviation)
    normalized <- dev_sum / max_val
    list(
      avg = mean(normalized),
      cont = as.data.frame(sort(normalized)) |> `colnames<-`("score"),
      t = tab
    )
  }
  x_v_y <- helper(x, y)
  y_v_x <- helper(y, x)
  avg <- mean(c(x_v_y$avg, y_v_x$avg))
  list(avg = avg, x_cols = x_v_y$cont, y_cols = y_v_x$cont, table = x_v_y$t)
}

confounding_score_multi <- function(df, cols) {
  pairs <- combn(cols, 2)
  tables <- list()
  between_pairs <- apply(pairs, 2, \(x) {
    name <- glue("{x[1]}_{x[2]}")
    res <- confounding_score(df, x[1], x[2])
    val <- list()
    val[[name]] <- res
    tables[[name]] <<- res$table
    val
  }) |>
    purrr::list_flatten()

  list(
    avg = mean(map_dbl(between_pairs, \(x) x$avg)),
    pairs = between_pairs,
    tables = tables
  )
}

# Very low confounding_score as expected
## test <- data.frame(
##   A = sample(c("a", "b", "c"), 1000, replace = TRUE),
##   B = sample(c(1, 2, 3), 1000, replace = TRUE)
## )

get_biosample_attrs <- function(ids, file = NULL) {
  if (is.null(file)) {
    concat <- paste0(ids, collapse = ",")
    args <- c("-db", "biosample", "-id", concat, "-format", "native")
    call <- system2("efetch", args, stdout = TRUE)
  } else {
    call <- read_lines(file)
  }
  i <- 0
  entry_indices <- map_dbl(call, \(x) {
    if (str_detect(x, "^[0-9]+: ")) {
      i <<- i + 1
      0
    } else {
      i
    }
  })
  entries <- split(call, entry_indices) |> discard_at(\(x) x == "0")
  lapply(entries, \(para) {
    find_acc <- keep(para, \(x) str_detect(x, "^Accession:"))
    acc <- str_extract(find_acc, "^Accession: (.*)\tID: [0-9]+$", group = 1)
    attributes <- str_trim(keep(
      para,
      \(x) str_detect(str_trim(x), "^/.*=\".*\"")
    ))
    a_names <- str_extract(attributes, "^/(.*)=.*", group = 1) |>
      str_replace_all(" ", "_")
    a_values <- str_extract(attributes, ".*=\"(.*)\"", group = 1)
    to_list <- as.list(a_values) |> `names<-`(a_names)
    tb <- tryCatch(expr = as_tibble(to_list), error = \(cnd) {
      warning(glue(
        "Detected duplicated column names in the following section\n{names(to_list)}"
      ))
      print(para)
      as_tibble(to_list, .name_repair = "unique")
    })
    mutate(tb, acc = acc)
  }) |>
    bind_rows()
}

get_bioproject_titles <- function(ids) {
  concat <- paste0(ids, collapse = ",")
  args <- c("-db", "bioproject", "-id", concat, "-format", "native")
  call <- system2("efetch", args, stdout = TRUE)
  i <- 1
  entry_indices <- map_dbl(call, \(x) {
    if (x == "") {
      i <<- i + 1
      0
    } else {
      i
    }
  })
  entries <- split(call, entry_indices) |> discard_at(\(x) x == "0")
  lapply(entries, \(para) {
    title <- str_remove(para[1], "^[0-9]+\\. ")
    find_acc <- keep(para, \(x) str_detect(x, "^BioProject Accession:"))
    acc <- str_remove(find_acc, "^BioProject Accession: ")
    org_mask <- str_detect(para, "^Organism: ")
    if (any(org_mask)) {
      organism <- str_remove(para[org_mask], "^Organism: ")
    } else {
      organism <- NA
    }
    tibble(acc = acc, title = title, organism = organism)
  }) |>
    bind_rows()
}

ncbi_taxid2rank <- function(taxids, rank = "kingdom", as_name = TRUE) {
  lapply(classification(unique(taxids), db = "ncbi"), \(x) {
    lookup <- dplyr::filter(x, rank == !!rank)
    if (nrow(lookup) > 1) {
      print(lookup)
      stop("should be one")
    }
    nrows <- nrow(lookup)
    is_na <- is.na(nrows)
    if (is_na || nrows == 0) {
      NA
    } else if (as_name) {
      lookup$name
    } else {
      lookup$id
    }
  }) |>
    unlist()
}

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

read_goa <- function(file) {
  read_tsv(
    file,
    col_names = c(
      "DB", # Database from which annotated entity has been taken
      "DB_Object_ID", # unique identifier in the database for the item being annotated
      "DB_Object_Symbol", # A unique and valid symbol (gene name) that corresponds to the DB_Object_ID
      # Official gene name is used if available
      "Qualifier", #  used for flags that modify the interpretation of an annotation
      # Possible options: NOT, colocalizes_with, contributes_to, NOT|contributes_to, NOT|colocalizes_with
      "GO_ID",
      "DB_Reference", # Reference cited to support an annotation, either citation
      # or GO_REF identifier
      "Evidence_Code", # one of the evidence codes supplied by the GO
      "With_or_From", # Additional identifier(s) to support annotations using
      # certain evidence codes (including IEA, IPI, IGI, IMP, IC and ISS evidences)
      "Aspect", # GO ontology code: P (biological process), F (molecular function) or C (cellular component)
      "DB_Object_Name", # full UniProt protein name will be present here, if available from UniProtKB
      "DB_Object_Synonym", # Alternative gene symbol(s) or
      # UniProtKB identifiers are provided pipe-separated, if available
      "DB_Object_Type", # The kind of entity being annotated, which for these files is 'protein'.
      "Taxon", # Identifier for the species being annotated or the gene product being defined.
      # An interacting taxon ID may be included in this column using a pipe to separate it from the primary taxon ID.
      "Date", # The date of last annotation update in the format 'YYYYMMDD'
      "Assigned_By", # Attribution for the source of the annotation
      "Annotation_Extension", # Contains cross references to other ontologies/databases
      # that can be used to qualify or enhance the GO term applied in the annotation.
      "Gene_Product_FormID" # The unique identifier of a specific spliceform of the DB_Object_ID.
    ),
    comment = "!"
  )
}

download_uniprot_fasta <- function(ids, outfile) {
  fastas <- lapply(ids, \(x) {
    system2(
      "curl",
      stdout = TRUE,
      args = glue("https://rest.uniprot.org/uniprotkb/{x}?format=fasta")
    )
  }) |>
    unlist()
  write_lines(fastas, outfile)
}

get_uniprot_confidence <- function(ids) {
  get_one <- function(id) {
    lst <- jsonlite::fromJSON(system2(
      "curl",
      stdout = TRUE,
      args = glue("https://rest.uniprot.org/uniprotkb/G8B4P6?format=json")
    ))
    tibble(
      acc = id,
      annotation_score = lst$annotationScore,
      protein_existence = lst$proteinExistence
    )
  }
  lapply(ids, get_one) |> bind_rows()
}
