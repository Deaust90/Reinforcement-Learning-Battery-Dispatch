# Battery environment 
battery_env <- new.env(parent = emptyenv())

create_env <- function(prices_vector) {
  env <- new.env(parent = emptyenv())
  reset_env(env, prices_vector)
  return(env)
}

# function to reset the env
reset_env <- function(env, prices_vector) {
  env$prices <- prices_vector
  env$time_index <- 1
  env$soc <- 0 # State of charge - reset at 0 
  env$total_profit <- 0
  env$done <- FALSE
}

# This function creates the state vector the agent expects.
# It takes the full price history, the current time step 't', and the current SoC.
make_state <- function(prices, t, soc, window_size) {
  # Get the window of past prices. Note the indices.
  price_window <- prices[(t - window_size + 1):t]
  
  # Combine the price window and the current state of charge
  state <- c(price_window, soc)
  
  return(state)
}

# Step function
step_env <- function(env, action) {
  if (env$done) stop("Game over. Reset the environment")
  
  price <- env$prices[env$time_index]
  energy <- POWER_LIMIT_MW15   #Constraint: This is a hardware limit 
  reward <- 0
  
  if (action == 1) {  # Charge
    charge_possible <- min(1 - env$soc, energy * EFFICIENCY)
    # Constraint: SOC cannot exceed 1 (battery full)
    # Constraint: Efficiency 90% → Only 90% of bought energy is stored
    net_energy <- charge_possible / EFFICIENCY  # Constraint: You pay for more energy than you store
    env$soc <- env$soc + charge_possible # Update SOC (limited above to not go > 1 due to min)
    reward <- -net_energy * price - net_energy * DEGRADATION_COST_EUR_MWH # Constraint: Pay for electricity + degradation cost (20 EUR/MWh throughput)
    
  } else if (action == 2) {  # Discharge
    discharge_possible <- min(env$soc, energy) # Constraint: Can’t discharge more than what is stored
    net_energy <- discharge_possible * EFFICIENCY # Constraint: Efficiency 90% → you only recover 90% of what you discharge
    env$soc <- env$soc - discharge_possible # Update SOC (won’t go below 0 due to min)
    reward <- net_energy * price - discharge_possible * DEGRADATION_COST_EUR_MWH # Constraint: Revenue from selling energy - degradation cost
  }
  
  env$total_profit <- env$total_profit + reward
  env$time_index <- env$time_index + 1
  
  if (env$time_index > length(env$prices)) {
    env$done <- TRUE # Environment ends when price series ends
  }
  
  return(list(
    reward = reward,
    done = env$done
  ))
}