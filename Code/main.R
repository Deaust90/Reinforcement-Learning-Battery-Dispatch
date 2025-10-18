# =================================================================
# MAIN TRAINING SCRIPT
# =================================================================

## 1. SETUP: LOAD LIBRARIES AND ALL CODE MODULES
# -----------------------------------------------------------------
library(keras)
library(tensorflow)
library(tidyverse)

# Source all the necessary scripts
source("config.R")         # Hyper-parameters from Person C
source("data_loader.R")   # Data loading from Person C
source("agent_logic.R")     # Agent functions from Person B
source("battery_settings.R")   # Environment from Person A

cat("--- All scripts loaded. Starting training setup. ---\n")

## 2. INITIALIZE AGENT, BUFFER, AND ENVIRONMENT
# -----------------------------------------------------------------
# Use the price vector from the data loader script
prices_raw <- price_train[1:15000] # Using the raw prices for the environment logic
n_steps <- length(prices_raw)

# Initialize agent's components
initialize_buffer(state_size = state_size, capacity = replay_capacity)
q_network <- build_q_network(state_size = state_size, action_size = action_size)
target_network <- build_q_network(state_size = state_size, action_size = action_size)
target_network$set_weights(q_network$get_weights())

# Initialize the battery environment from Person A's script
reset_env(battery_env, prices_raw)

cat("--- Agent and Environment are ready. ---\n")

## 3. MAIN TRAINING LOOP
# -----------------------------------------------------------------

# Calculate the decay rate
eps_decay_rate <- log(eps_start / eps_final) / (episodes - 1)

# Epsilon function
epsilon <- function(ep) {
  eps_start * exp(-eps_decay_rate * (ep - 1))
}

for (ep in 1:episodes) {
  eps_current <- epsilon(ep)
  # Reset the battery environment and profit for the new episode
  reset_env(battery_env, prices_raw)
  
  # The step loop iterates through the price data
  # We start at 'window_size' because we need a full history for the first state
  for (t in window_size:(n_steps - 1)) {
    
    # A. GET CURRENT STATE
    # Use our new function to get the state in the correct format
    # Use the NORMALIZED prices to create the state for the neural network
    current_state <- make_state(prices = prices_norm, t = t, soc = battery_env$soc, window_size = window_size)
    
    # B. CHOOSE AN ACTION
    # Your agent's logic for selecting an action (epsilon-greedy)
    action <- agent_act(model = q_network, state = current_state, epsilon = eps_current)
    
    # C. INTERACT WITH THE ENVIRONMENT
    # The environment processes the action and gives back the result
    result <- step_env(battery_env, action)
    
    # D. OBSERVE AND STORE
    # Get the state AFTER the action was taken
    # Use the NORMALIZED prices again for the next state
    next_state <- make_state(prices = prices_norm, t = (t + 1), soc = battery_env$soc, window_size = window_size)
    
    # Store this entire experience in your replay buffer
    store_transition(
      s = current_state,
      a = action,
      r = result$reward,
      s2 = next_state,
      done = result$done
    )
    
    # E. LEARN
    # If the buffer has enough memories, train the agent
    if (replay$idx > warmup_mem || replay$full) {
      agent_learn(q_net = q_network, target_net = target_network, batch_size = batch_size)
    }
    
    # If the episode is over, break the inner loop
    if (result$done) break
  }
  
  # F. SYNCHRONIZE NETWORKS
  # At the end of the episode, update the target network's weights
  if (ep %% target_sync_freq == 0) {
    target_network$set_weights(q_network$get_weights())
    cat(sprintf("--- Target network updated at episode %d ---\n", ep))
  }
  gc()
  # Print episode summary
  cat(sprintf("Episode %d | Total Profit: %.2f\n", ep, battery_env$total_profit))
}

cat("\n--- TRAINING COMPLETE ---\n")

## 4. SAVE YOUR TRAINED MODEL
# -----------------------------------------------------------------
# After training, save the weights of your final q_network
# save_model_weights_hdf5(q_network, "dqn_battery_model.h5")

## 5. EVALUATION ON THE TEST DATA
source("evaluation.R")  # Results on the test data
