# get an array of publication lags of variables from the catalog
# 1 before, etc., mean data as it was available then. E.g. target date is 06-2020, 1 before means data as it was in 05-2020. So a variable with publication lag 1 would have data in that vintage until 03-2020, lag 2 would have until 02-2020
gen_lags <- function(data, catalog) {
  variable_lags <- catalog %>% filter(code %in% colnames(data)) %>% select(code, publication_lag) %>% 
    right_join(data.frame(code=colnames(data)[2:length(colnames(data))]),by="code") %>% select(publication_lag) %>% pull
  return (variable_lags)
}

evaluate_dfm <- function (data, target_variable, variable_lags, test_proportion=0.85, p=1, max_iter=1500, threshold=1e-5) {
  # split into test and train sets
  data <- data %>% arrange(date)
  test_row <- as.integer(nrow(data) * test_proportion)
  train <- data[1:(test_row-1),]
  
  # estimate dfm on training data
  output_dfm <- dfm(train, p=p, max_iter=max_iter, threshold=threshold)
  
  # function for lagging a series
  lag_series <- function(dates, series, n_lag, as_of_date) {
    series[(which(dates==as_of_date) - n_lag):length(series)] <- NA
    return (series)
  }
  
  # initializing RMSE and MAE
  RMSE <- MAE <- list()
  
  # calculating performance on vintage
  dates <- data$date
  tmp_RMSE_2_before <- tmp_RMSE_1_before <- tmp_RMSE_0_before <- tmp_RMSE_1_after <- tmp_RMSE_2_after <- c()
  tmp_MAE_2_before <- tmp_MAE_1_before <- tmp_MAE_0_before <- tmp_MAE_1_after <- tmp_MAE_2_after <- c()
  x2_before <- x1_before <- x0_before <- x1_after <- x2_after <- c()
  x2_before_date <- x1_before_date <- x0_before_date <- x1_after_date <- x2_after_date <- c()
  
  for (row in (test_row-2):(nrow(data)-2)) { # -2 to make sure there's enough data at the end
    date <- as.Date(data[row, "date"])
    vintage <- data %>% 
      filter(date <= !!date)
    for (col in 2:ncol(vintage)) {
      vintage[,col] <- lag_series(dates, vintage[,col], variable_lags[col-1], date)
    }
    pred <- predict_dfm(vintage, output_dfm, months_ahead=3)
    
    pred_2_before <- pred[row+2,target_variable]; pred_2_before_date <- pred[row+2,"date"]
    pred_1_before <- pred[row+1,target_variable]; pred_1_before_date <- pred[row+1,"date"]
    pred_0_before<- pred[row+0,target_variable]; pred_0_before_date <- pred[row+0,"date"]
    pred_1_after <- pred[row-1,target_variable]; pred_1_after_date <- pred[row-1,"date"]
    pred_2_after <- pred[row-2,target_variable]; pred_2_after_date <- pred[row-2,"date"]
    
    x2_before <- append(x2_before, pred_2_before); x2_before_date <- append(x2_before_date, pred_2_before_date)
    x1_before <- append(x1_before, pred_1_before); x1_before_date <- append(x1_before_date, pred_1_before_date)
    x0_before <- append(x0_before, pred_0_before); x0_before_date <- append(x0_before_date, pred_0_before_date)
    x1_after <- append(x1_after, pred_1_after); x1_after_date <- append(x1_after_date, pred_1_after_date)
    x2_after <- append(x2_after, pred_2_after); x2_after_date <- append(x2_after_date, pred_2_after_date)
    
    tmp_RMSE_2_before <- append(tmp_RMSE_2_before, ((pred_2_before) - data[data$date == data[row+2, "date"], target_variable])^2)
    tmp_RMSE_1_before <- append(tmp_RMSE_1_before, ((pred_1_before) - data[data$date == data[row+1, "date"], target_variable])^2)
    tmp_RMSE_0_before <- append(tmp_RMSE_0_before, ((pred_0_before) - data[data$date == data[row+0, "date"], target_variable])^2)
    tmp_RMSE_1_after <- append(tmp_RMSE_1_after, ((pred_1_after) - data[data$date == data[row-1, "date"], target_variable])^2)
    tmp_RMSE_2_after <- append(tmp_RMSE_2_after, ((pred_2_after) - data[data$date == data[row-2, "date"], target_variable])^2)
    
    tmp_MAE_2_before <- append(tmp_MAE_2_before, (abs(data[data$date == data[row+2, "date"], target_variable] - pred_2_before)))
    tmp_MAE_1_before <- append(tmp_MAE_1_before, (abs(data[data$date == data[row+1, "date"], target_variable] - pred_1_before)))
    tmp_MAE_0_before <- append(tmp_MAE_0_before, (abs(data[data$date == data[row+0, "date"], target_variable] - pred_0_before)))
    tmp_MAE_1_after <- append(tmp_MAE_1_after, (abs(data[data$date == data[row-1, "date"], target_variable] - pred_1_after)))
    tmp_MAE_2_after <- append(tmp_MAE_2_after, (abs(data[data$date == data[row-2, "date"], target_variable] - pred_2_after)))
  }
  
  pred_df <- data.frame(date=data$date, actual=data[,target_variable]) %>% 
    left_join(data.frame(date=x2_before_date, x2_before=x2_before), by="date") %>% 
    left_join(data.frame(date=x1_before_date, x1_before=x1_before), by="date") %>% 
    left_join(data.frame(date=x0_before_date, x0_before=x0_before), by="date") %>% 
    left_join(data.frame(date=x1_after_date, x1_after=x1_after), by="date") %>% 
    left_join(data.frame(date=x2_after_date, x2_after=x2_after), by="date")
  
  # filtering for only those periods where every lag had a prediction
  min_date <- min(x2_before_date)
  max_date <- max(x2_after_date)
  
  RMSE[["-2"]] <- sqrt(mean(tmp_RMSE_2_before[x2_before_date >= min_date & x2_before_date <= max_date], na.rm=TRUE))
  RMSE[["-1"]] <- sqrt(mean(tmp_RMSE_1_before[x1_before_date >= min_date & x1_before_date <= max_date], na.rm=TRUE))
  RMSE[["0"]] <- sqrt(mean(tmp_RMSE_0_before[x0_before_date >= min_date & x0_before_date <= max_date], na.rm=TRUE))
  RMSE[["+1"]] <- sqrt(mean(tmp_RMSE_1_after[x1_after_date >= min_date & x1_after_date <= max_date], na.rm=TRUE))
  RMSE[["+2"]] <- sqrt(mean(tmp_RMSE_2_after[x2_after_date >= min_date & x2_after_date <= max_date], na.rm=TRUE))
  
  MAE[["-2"]] <- mean(tmp_MAE_2_before[x2_before_date >= min_date & x2_before_date <= max_date], na.rm=TRUE)
  MAE[["-1"]] <- mean(tmp_MAE_1_before[x1_before_date >= min_date & x1_before_date <= max_date], na.rm=TRUE)
  MAE[["0"]] <- mean(tmp_MAE_0_before[x0_before_date >= min_date & x0_before_date <= max_date], na.rm=TRUE)
  MAE[["+1"]] <- mean(tmp_MAE_1_after[x1_after_date >= min_date & x1_after_date <= max_date], na.rm=TRUE)
  MAE[["+2"]] <- mean(tmp_MAE_2_after[x2_after_date >= min_date & x2_after_date <= max_date], na.rm=TRUE)
  
  # AIC and BIC
  # Calculate number of parameters
  num_param <- 
    sum(output_dfm$blocks) +           # Parameters in the measurement equation
    ncol(output_dfm$blocks) * p +      # Lag parameters in the transition equation of the factors
    ncol(output_dfm$Xsmooth) * p +     # Lag parameters in the transition equation of the idiosyncratic terms
    ncol(output_dfm$blocks) +          # Variances of error term of factors in the transition equation
    ncol(output_dfm$Xsmooth)           # Variances of error term of idiosyncratic terms in the transition equation
  AIC <- 2 * num_param - 2 * output_dfm$loglik
  BIC <- num_param * log(nrow(output_dfm$Xsmooth)) - 2 * output_dfm$loglik
  
  return(list(
    pred_df=pred_df,
    AIC=AIC[1],
    BIC=BIC[1],
    RMSE=RMSE,
    MAE=MAE
  ))
}