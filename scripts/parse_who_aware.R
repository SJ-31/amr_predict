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
  group <- xml_find_all(node, ".//span[@class]") |>
    xml_text() |>
    str_remove(" group")
  tibble(name = name, group = group, indications = indications)
}) |>
  bind_rows()
