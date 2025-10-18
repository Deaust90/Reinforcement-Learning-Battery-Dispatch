library(keras)
library(tidyverse)
library(lubridate)

## LOAD & PREPARE DATA -------------------------------------------------
data_task4 <- read_csv("data_task4.csv", col_names = TRUE) %>%
  rename(
    datetime_utc = `DateTime(UTC)`,
    price        = `Price[Currency/MWh]`
  )

# Convert timezone and sort chronologically
data_task4 <- data_task4 %>%
  mutate(
    datetime_utc = with_tz(datetime_utc, tzone = "UTC"),
    datetime_cet = with_tz(datetime_utc, tzone = "Europe/Vienna")
  ) %>%
  arrange(datetime_cet)

# Extract year for splitting
data_task4 <- data_task4 %>%
  mutate(year = year(datetime_cet))

anyNA(data_task4)

## TRAIN–TEST SPLIT -----------------------------------------------------

# Split by year (Train: 2019–2022, Test: 2023–2024)
train_data <- data_task4 %>% filter(year >= 2019 & year <= 2022)
test_data  <- data_task4 %>% filter(year >= 2023)

# Extract price vectors for RL
price_train <- train_data$price
price_test  <- test_data$price

cat("Train samples:", length(price_train), "\n")
cat("Test samples :", length(price_test), "\n")

head(train_data)

# Calculate data-driven thresholds ONLY from training data
charge_threshold    <- quantile(train_data$price, 0.20)
discharge_threshold <- quantile(train_data$price, 0.80)

# For training, we use the price vector from the training data
n_train <- length(price_train)
max_steps_ep <- n_train - window_size - 1

n_test <- length(price_test)

# Normalize prices for stable network training
price_mean <- mean(price_train)
price_sd   <- sd(price_train)
prices_norm <- (price_train - price_mean) / price_sd

## BENCHMARK POLICY IMPLLEMENTATION ------------------------------------

# T2(a): Random Policy
random_policy <- function(SoC_mwh, price) {
  # Ignores inputs and chooses a random action
  return(sample(0:2, 1))
}

# T2(b): Rule-Based Policy
rule_based_policy <- function(SoC_mwh, price) {
  if (price < charge_threshold) {
    return(1) # Action: Charge
  } else if (price > discharge_threshold) {
    return(2) # Action: Discharge
  } else {
    return(0) # Action: Hold
  }
}


## EVALUATION FUNCTIONS ------------------------------------------------

# A generic function to run any policy on the test data and get the result.
evaluate_policy <- function(policy_function, prices_vector, env) {
  
  # Reset the passed environment properly
  reset_env(env, prices_vector)
  
  n_steps <- length(prices_vector)
  
  soc_history <- numeric(n_steps)
  action_history <- integer(n_steps)
  
  for (t in 1:n_steps) {
    if (env$done) break
    
    current_price <- prices_vector[t]
    action <- policy_function(env$soc, current_price)
    
    result <- step_env(env, action)
    
    soc_history[t] <- env$soc
    action_history[t] <- action
  }
  
  return(list(
    final_profit = env$total_profit,
    soc_history = soc_history,
    action_history = action_history
  ))
}

evaluate_dqn_agent <- function(q_network, prices_raw, prices_normalized) {  
  # Use the existing battery_env
  reset_env(battery_env, prices_raw)
  n_steps <- length(prices_raw)
  
  # History trackers
  soc_history <- numeric(n_steps)
  action_history <- integer(n_steps)
  profit_history <- numeric(n_steps)
  
  # Evaluation loop (greedy policy)
  for (t in window_size:(n_steps - 1)) {
    if (battery_env$done) break
    
    # 1. Get the state (normalized prices + current SOC)
    current_state <- make_state(prices = prices_normalized, t = t, soc = battery_env$soc, window_size = window_size)
    
    # 2. Greedy action (epsilon = 0)
    action <- agent_act(q_network, current_state, epsilon = 0)
    
    # 3. Step the environment
    result <- step_env(battery_env, action)
    
    # 4. Save histories
    soc_history[t] <- battery_env$soc
    action_history[t] <- action
    profit_history[t] <- battery_env$total_profit
  }
  
  return(list(
    final_profit = battery_env$total_profit,
    soc_history = soc_history,
    action_history = action_history,
    profit_history = profit_history
  ))
}