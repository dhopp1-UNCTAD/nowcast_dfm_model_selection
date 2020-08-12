library(tidyverse)
options(scipen=999)

convert_monthly <- function(dates, x) {
  new <- rep(NA, length(x))
  for (i in 3:length(x)) {
    if (substr(dates[i], 6, 7) %in% c("03", "06", "09", "12") & !anyNA(x[(i-2):i])) {
      new[i] <- prod(x[(i-2):i] + 1) - 1
    }
  }
  return (new)
}

evaluate_variable <- function(dates, y, x) {
  # convert monthly series to quarterly, only if target variable is quarterly
  if (length(sapply(dates[!is.na(y)], function(x) substr(x, 6, 7)) %>% unique) == 4) {
    if ((setdiff((sapply(dates[!is.na(x)], function(x) substr(x, 6, 7)) %>% unique), c("03", "06", "09", "12")) %>% length) > 0) {
      x <- convert_monthly(dates, x)
    }
  }
  y_na <- !is.na(y)
  x_na <- !is.na(x)
  dates <- dates[y_na & x_na]
  x <- x[y_na & x_na]
  y <- y[y_na & x_na]
  
  r2 <- summary(lm(formula = y ~ x))$r.squared
  correlation <- cor(x, y)
  n_values <- length(y)
  
  return (list(
    r2=r2,
    correlation=correlation,
    n_values=n_values
  ))
}

elapsed_months <- function(end_date, start_date) {
  ed <- as.POSIXlt(end_date)
  sd <- as.POSIXlt(start_date)
  12 * (ed$year - sd$year) + (ed$mon - sd$mon)
}

elapsed_months(as.Date("2018-02-01"), as.Date("2020-01-01"))

# evaluating
data <- read_csv("/home/danhopp/dhopp1/UNCTAD/nowcast_data_update/output/2020-08-11_database_tf.csv") %>% data.frame
catalog <- read_csv("/home/danhopp/dhopp1/UNCTAD/nowcast_data_update/helper/catalog.csv") %>% data.frame

tmp <- data.frame(target_variable=character(), variable=character(), r2=numeric(), correlation=numeric(), n_values=numeric(), publication_lag=numeric(), start_date=numeric())
for (target_variable in c("x_world", "x_vol_world2", "x_servs_world")) {
   for (col in colnames(data)) {
     if (!(col %in% c("date", target_variable))) {
       publication_lag <- catalog %>% filter(code==col) %>% select(publication_lag) %>% pull
       start_date <- max(elapsed_months(catalog %>% filter(code==col) %>% mutate(date = as.Date(start_date, format="%m/%d/%Y")) %>% select(date) %>% pull, min(data$date)), 0)
       eval <- evaluate_variable(data$date, data[,target_variable], data[,col])
       r2 <- eval$r2
       correlation <- eval$correlation
       n_values <- eval$n_values
       
       tmpi <- data.frame(target_variable=target_variable, variable=col, r2=r2, correlation=correlation, n_values=n_values, publication_lag=publication_lag, start_date=start_date)
       tmp <- rbind(tmp, tmpi)
     }
   }
}
write_csv(tmp, "variable_ranking.csv")