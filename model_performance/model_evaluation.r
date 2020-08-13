library(tidyverse)
library(nowcastDFM)
Sys.setlocale(category = "LC_NUMERIC", locale = "en_US.UTF-8")
source("evaluate_dfm.r")

ranking <- read_csv("../variable_ranking/norm_variable_ranking.csv")
data <- read_csv("/home/danhopp/dhopp1/UNCTAD/nowcast_data_update/output/2020-08-11_database_tf.csv")
catalog <- read_csv("/home/danhopp/dhopp1/UNCTAD/nowcast_data_update/helper/catalog.csv")

if (T) {
  results <- read_csv("results.csv") 
} else {
  names <- c("model_num", "target_variable", "p", "AIC", "BIC", "RMSE_2_back", "RMSE_1_back", "RMSE_0_back", "RMSE_1_ahead", "RMSE_2_ahead", "RMSE", "MAE_2_back", "MAE_1_back", "MAE_0_back", "MAE_1_ahead", "MAE_2_ahead", "MAE", "min_rank", "max_rank", "mean_rank", "median_rank", unique(catalog$code))
  results <- setNames(data.frame(matrix(ncol = length(names), nrow = 0)), names)
}
last_result_row <- nrow(results)

# model loop
for (target_variable in c("x_world", "x_vol_world2", "x_servs_world")) {
  for (i in 1:500) {
    print(paste0("Running model ", i, " ", target_variable))
    # parameters
    #target_variable <- "x_vol_world2"
    n_vars <- as.integer(runif(1, 5, 30)) # random number of variables
    which_slice <- unique(as.integer(runif(n_vars, 1, 100))) # random slice amongst the top 100
    p <- 1
    which_blocks <- c("Block1-Global") # columns in catalog
    
    vars <- ranking %>% 
      filter(target_variable == !!target_variable) %>% 
      slice(which_slice) %>% 
      select(variable) %>% pull
    model_data <- data[,c("date", target_variable, vars)] %>% data.frame
    blocks <- data.frame(code=vars) %>% left_join(catalog, by="code") %>% select(which_blocks) %>% data.frame
    
    variable_lags <- gen_lags(model_data, catalog)
    status <- tryCatch({
      model <- evaluate_dfm(model_data, target_variable, variable_lags, test_proportion=0.85, blocks=blocks, p=p, max_iter=1500, threshold=1e-5)
      TRUE
    }, error = function(e) { FALSE })
    
    results[last_result_row + 1, "model_num"] <- last_result_row + 1
    results[last_result_row + 1, "target_variable"] <- target_variable
    results[last_result_row + 1, "p"] <- p
    results[last_result_row + 1, "target_variable"] <- target_variable
    # block represented as 1-0-0, e.g. if there are three blocks and variable is in 1st and 3rd, 1-0-1. If only one block will be 1 or 0.
    results[last_result_row + 1, unique(catalog$code)] <- paste(rep("0", ncol(blocks)), collapse="-")
    counter <- 1
    for (variable in vars) {
      results[last_result_row + 1, variable] <- paste(as.character(blocks[counter,]), collapse="-")
      counter <- counter + 1
    }
    tmp_ranking <- ranking %>%  filter(target_variable == !!target_variable) %>% mutate(rank=1:n()) %>% filter(variable %in% vars)
    results[last_result_row + 1, "min_rank"] <- min(tmp_ranking$rank)
    results[last_result_row + 1, "max_rank"] <- max(tmp_ranking$rank)
    results[last_result_row + 1, "mean_rank"] <- mean(tmp_ranking$rank)
    results[last_result_row + 1, "median_rank"] <- median(tmp_ranking$rank)
    
    # performance criteria will be NA if the model can't run
    if (status) {
      results[last_result_row + 1, c("AIC", "BIC", "RMSE_2_back", "RMSE_1_back", "RMSE_0_back", "RMSE_1_ahead", "RMSE_2_ahead", "MAE_2_back", "MAE_1_back", "MAE_0_back", "MAE_1_ahead", "MAE_2_ahead")]
      results[last_result_row + 1, "AIC"] <- model$AIC
      results[last_result_row + 1, "BIC"] <- model$BIC
      results[last_result_row + 1, "RMSE_2_back"] <- model$RMSE[["-2"]]
      results[last_result_row + 1, "RMSE_1_back"] <- model$RMSE[["-1"]]
      results[last_result_row + 1, "RMSE_0_back"] <- model$RMSE[["0"]]
      results[last_result_row + 1, "RMSE_1_ahead"] <- model$RMSE[["+1"]]
      results[last_result_row + 1, "RMSE_2_ahead"] <- model$RMSE[["+2"]]
      results[last_result_row + 1, "RMSE"] <- mean(unlist(model$RMSE))
      results[last_result_row + 1, "MAE_2_back"] <- model$MAE[["-2"]]
      results[last_result_row + 1, "MAE_1_back"] <- model$MAE[["-1"]]
      results[last_result_row + 1, "MAE_0_back"] <- model$MAE[["0"]]
      results[last_result_row + 1, "MAE_1_ahead"] <- model$MAE[["+1"]]
      results[last_result_row + 1, "MAE_2_ahead"] <- model$MAE[["+2"]]
      results[last_result_row + 1, "MAE"] <- mean(unlist(model$MAE))
    }
    last_result_row <- last_result_row + 1
    
    write_csv(results, "results.csv")
  }
}
# best model, 14! for x_world
best_model <- results %>% filter(target_variable == !!target_variable) %>% mutate(final = RMSE * as.numeric(MAE)) %>% arrange(final) %>% slice(1:1) %>% select(model_num) %>% pull
best_vars <- colnames(results)[(which(colnames(results) == "median_rank")+1):ncol(results)][results[best_model, (which(colnames(results) == "median_rank")+1):ncol(results)] == 1]
