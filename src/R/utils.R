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
      cont = as.data.frame(sort(normalized)) |> `colnames<-`("score")
    )
  }
  x_v_y <- helper(x, y)
  y_v_x <- helper(y, x)
  avg <- mean(c(x_v_y$avg, y_v_x$avg))
  list(avg = avg, x_cols = x_v_y$cont, y_cols = y_v_x$cont)
}

confounding_score_multi <- function(df, cols) {
  pairs <- combn(cols, 2)
  between_pairs <- apply(pairs, 2, \(x) {
    name <- glue("{x[1]}_{x[2]}")
    res <- confounding_score(df, x[1], x[2])
    val <- list()
    val[[name]] <- res
    val
  }) |>
    purrr::list_flatten()

  list(
    avg = mean(map_dbl(between_pairs, \(x) x$avg)),
    pairs = between_pairs
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
