library(tidyverse)
library(nowcastDFM)
library(stringr)
Sys.setlocale(category = "LC_NUMERIC", locale = "en_US.UTF-8")
source("evaluate_dfm.r")
source("model_evaluation_helper.r")

# reading in necessary CSVs
helper_directory <- "/home/danhopp/dhopp1/UNCTAD/nowcast_data_update/helper/"
output_directory <- "/home/danhopp/dhopp1/UNCTAD/nowcast_data_update/output/"
database <- "2020-08-07_database_tf.csv"
ranking <- read_csv("../variable_ranking/norm_variable_ranking.csv")
data <- read_csv(paste0(output_directory, database))
catalog <- read_csv(paste0(helper_directory, "catalog.csv"))
if (T) {
  results <- read_csv("results.csv") 
} else {
  names <- c("model_num", "target_variable", "p", "AIC", "BIC", "RMSE_2_back", "RMSE_1_back", "RMSE_0_back", "RMSE_1_ahead", "RMSE_2_ahead", "RMSE", "MAE_2_back", "MAE_1_back", "MAE_0_back", "MAE_1_ahead", "MAE_2_ahead", "MAE", "min_rank", "max_rank", "mean_rank", "median_rank", unique(catalog$code))
  results <- setNames(data.frame(matrix(ncol = length(names), nrow = 0)), names)
}


# model loop for many random. 08.09 running again for x_world, keeping out variables don't want. 1734 on is this new run
for (target_variable in c("x_world", "x_vol_world2", "x_servs_world")) {
  for (i in 1:500) {
    print(paste0("Running model ", i, " ", target_variable))
    # parameters
    n_vars <- as.integer(runif(1, 5, 30)) # random number of variables
    which_slice <- unique(as.integer(runif(n_vars, 1, 150))) # random slice amongst the top 100
    ### selecting out bad variables don't want in the run
    which_rank <- ranking %>% filter(target_variable=="x_world") %>% mutate(rank=1:n())
    bad_vars <- c("x_cn_2", # variables I don't want in there, eikon etc.
                  "constr_cn",
                  "freight_mar_cn",
                  "freight_mar_cn",
                  "ipi_cn",
                  "rti_cn",
                  "pmi_manuf_cn",
                  "export_orders_cn",
                  "pmi_nm_cn2",
                  "pmi_total_cn",
                  "is_cn",
                  "fc_gdp_de",
                  "x_servs_uk2",
                  "fdi_cn",
                  "fdi_in",
                  "baltic",
                  "harpex",
                  "pmi_manuf_us",
                  "pmi_nm_us",
                  "pmi_nm_cn",
                  "oil_x",
                  "oil_prod",
                  "tourists_th",
                  "fc_gdp_de",            
                  "fc_gdp_jp", 
                  "fc_gdp_oecd", 
                  "fc_gdp_uk", 
                  "fc_gdp_us",
                  "x_mx")
    bad_vars_nums <- which_rank %>% filter(variable %in% bad_vars) %>% select(rank) %>% pull
    which_slice <- which_slice[!(which_slice %in% bad_vars_nums)]
    ### selecting out bad variables don't want in the run
    p <- 1
    which_blocks <- c("Block1-Global") # columns in catalog
    results <- run_single(results, ranking, data, target_variable, which_slice=which_slice, which_blocks=which_blocks, p=p, write_output=TRUE)
  }
}


# best model experimentation
# x_world = 1733 is best w/o fc_gdp_de
# x_vol_world2 = 1703
# x_servs_world = 1704 (without eikon)
target_variable <- "x_world"
p <- 1
best_model <- get_best_model(results, target_variable, rank=1)
which_slice <- get_slice(results, ranking, best_model)$slices
which_blocks <- c("Block1-Global")
results <- run_single(results, ranking, data, target_variable, which_slice, which_blocks, p, write_output=FALSE)
derived_best <- 1731 # when run again get slightly different numbers, compare to this one
compare_models(results, derived_best, nrow(results))

#results %>% mutate(wow=ifelse(model_num==nrow(results),"yes","no")) %>% filter(target_variable==!!target_variable) %>% mutate(final=as.numeric(RMSE) * as.numeric(MAE)) %>% arrange(final) %>% mutate(rank=1:n()) %>% 
#  ggplot() + aes(x=rank, y=final, color=wow) + geom_point()

# write best into blocks in catalog
best_x_world <- 1902
best_x_vol_world2 <- 1722
best_x_servs_world <- 1704

catalog$x_world_p <- results %>% filter(model_num == best_x_world) %>% select(p) %>% pull
catalog$x_vol_world2_p <- results %>% filter(model_num == best_x_vol_world2) %>% select(p) %>% pull
catalog$x_servs_world_p <- results %>% filter(model_num == best_x_servs_world) %>% select(p) %>% pull

for (row in 1:nrow(catalog)) {
  catalog[row, c("x_world", "x_world_block_1")] <- results %>% filter(model_num==best_x_world) %>% select(catalog[row,"code"] %>% pull) %>% pull %>% as.numeric
}
for (row in 1:nrow(catalog)) {
  catalog[row, c("x_vol_world2", "x_vol_world2_block_1")] <- results %>% filter(model_num==best_x_vol_world2) %>% select(catalog[row,"code"] %>% pull) %>% pull %>% as.numeric
}
for (row in 1:nrow(catalog)) {
  catalog[row, c("x_servs_world", "x_servs_world_block_1")] <- results %>% filter(model_num==best_x_servs_world) %>% select(catalog[row,"code"] %>% pull) %>% pull %>% as.numeric
}
write_csv(catalog, paste0(helper_directory, "catalog.csv"))