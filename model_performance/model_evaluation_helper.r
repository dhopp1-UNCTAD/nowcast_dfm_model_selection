### helping functions
# function to run a single model
run_single <- function(results, ranking, data, target_variable, which_slice, which_blocks, p, write_output=FALSE) {
  last_result_row <- nrow(results)
  
  vars <- ranking %>% 
    filter(target_variable == !!target_variable) %>% 
    slice(which_slice) %>% 
    select(variable) %>% pull %>% as.character
  vars <- c(target_variable, vars) %>% unique
  var_orders <- 1:length(vars)
  model_data <- data[,c("date", vars)] %>% data.frame
  if (exists("catalog")) { # for generality, if catalog doesn't exist just do 1 global block
    blocks <- data.frame(code=vars) %>% left_join(catalog, by="code") %>% select(which_blocks) %>% data.frame 
  } else {
    blocks <- which_blocks
  }
  
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
    results[last_result_row + 1, variable] <- str_replace(paste(as.character(blocks[counter,]), collapse="-"), "1", as.character(counter)) # replace 1 with 
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
  if (write_output) {
    write_csv(results, "results.csv")
  }
  return (results)
}

# function to compare two model #s
compare_models <- function(results, model_num_1, model_num_2) {
  mod_1 <- results %>% filter(model_num==model_num_1) %>% mutate(BIC=as.numeric(BIC), AIC=as.numeric(AIC), RMSE=as.numeric(RMSE), MAE=as.numeric(MAE))
  mod_2 <- results %>% filter(model_num==model_num_2) %>% mutate(BIC=as.numeric(BIC), AIC=as.numeric(AIC), RMSE=as.numeric(RMSE), MAE=as.numeric(MAE))
  for (thing in c("BIC", "AIC", "RMSE", "MAE")) {
    print(str_interp("${thing} | ${model_num_1}: ${round(mod_1[,thing], 4)} | ${model_num_2}: ${round(mod_2[,thing], 4)} | ${model_num_1} - ${model_num_2} = ${mod_1[,thing] - mod_2[,thing]}, model ${if (mod_1[,thing] - mod_2[,thing] > 0){model_num_2}else{model_num_1}} better"))
  }
}

# get the variable slices of a model_number
get_slice <- function(results, ranking, model_num) {
  mod <- results %>% filter(model_num==!!model_num)
  first_col <- which(colnames(mod) == "median_rank")+1
  vars <- colnames(mod)[first_col:ncol(mod)][mod[,first_col:ncol(mod)] != 0]
  which_rank <- ranking %>% filter(target_variable==mod$target_variable[1]) %>% mutate(rank=1:n())
  ranks <- which_rank %>% filter(variable %in% vars) %>% select(rank) %>% pull
  return (list(vars=vars, slices=ranks))
}

# get model num of best performing model by MAE * RMSE
get_best_model <- function(results, target_variable, rank=1) {
  best_model <- results %>% filter(target_variable == !!target_variable) %>% mutate(final = as.numeric(RMSE) * as.numeric(MAE)) %>% arrange(final) %>% slice(rank:rank) %>% select(model_num) %>% pull
  return (best_model)
}
### helping functions