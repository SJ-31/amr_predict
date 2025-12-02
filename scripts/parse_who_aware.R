library(xml2)
library(tidyverse)
url <- "https://aware.essentialmeds.org/list"
file <- tempfile()
download_html(url, file = file)
root <- read_html(file)

antibiotics <- xml_find_all(root, "//div[@class='medicine-name']")

data <- xml_find_all(root, "//div[@class='medicine-details']")

test <- data[1]

aware <- lapply(data, \(node) {
  name <- xml_find_first(node, ".//a[@title]") |>
    xml_text() |>
    str_trim()
  indications <- xml_find_all(node, ".//span[@class='indication-text']") |>
    xml_text() |>
    paste0(collapse = ";")
  never_listed <- xml_find_all(node, ".//div[@class='never-listed-status']")
  group <- xml_find_all(node, ".//span[@class='group']") |>
    xml_text() |>
    str_remove(" group")
  tibble(
    name = name,
    group = group,
    indications = indications,
    essential = length(never_listed) == 0
  )
}) |>
  bind_rows() |>
  distinct()
