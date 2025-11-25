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

# Very confounding score low as expected
## test <- data.frame(
##   A = sample(c("a", "b", "c"), 1000, replace = TRUE),
##   B = sample(c(1, 2, 3), 1000, replace = TRUE)
## )
