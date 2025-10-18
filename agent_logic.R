## =================================================================
## REPLAY BUFFER
## =================================================================

replay <- new.env(parent = emptyenv())

# Initialize the buffer structure
initialize_buffer <- function(state_size, capacity = replay_capacity) {
  replay$S    <- array(0, dim = c(capacity, state_size))
  replay$A    <- integer(capacity)
  replay$R    <- numeric(capacity)
  replay$S2   <- array(0, dim = c(capacity, state_size))
  replay$D    <- integer(capacity)
  replay$idx  <- 1L
  replay$full <- FALSE
  cat("Replay Buffer initialized.\n")
}

# Store a single transition (s, a, r, s', done)
store_transition <- function(s, a, r, s2, done) {
  i <- replay$idx
  replay$S[i, ]  <- s
  replay$A[i]    <- a
  replay$R[i]    <- r
  replay$S2[i, ] <- s2
  replay$D[i]    <- as.integer(done) # Ensure done is integer (0 or 1)
  
  replay$idx <- i %% replay_capacity + 1L
  if (i == replay_capacity) replay$full <- TRUE
}

# Sample a batch of transitions for training
sample_batch <- function(size) {
  max_i <- if (replay$full) replay_capacity else (replay$idx - 1L)
  idx   <- sample.int(max_i, size)
  
  list(
    S  = replay$S[idx, , drop = FALSE],
    A  = replay$A[idx],
    R  = replay$R[idx],
    S2 = replay$S2[idx, , drop = FALSE],
    D  = replay$D[idx]
  )
}


## =================================================================
## BUILD Q-NETWORK + TARGET NETWORK
## =================================================================

build_q_network <- function(state_size, action_size) {
  input  <- layer_input(shape = state_size)   
  hidden <- input %>% 
    layer_dense(units = hidden_units, activation = "relu") %>%
    layer_dense(units = hidden_units, activation = "relu")
  
  # The output layer must have 'action_size' units (3 now)
  output <- hidden %>% layer_dense(units = action_size, activation = "linear") 
  
  model <- keras_model(input, output)
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = learning_rate), 
    loss = "mse"
  )
  
  return(model)
}


## =================================================================
## AGENT'S CORE LOGIC
## =================================================================

# --- Action Selection Function (Epsilon-Greedy) ---
agent_act <- function(model, state, epsilon) {
  if (runif(1) < epsilon) {
    return(sample(0:(action_size - 1), 1)) 
  }
  
  # The state from the environment is an R vector.
  # Convert it to a tensor before giving it to the model.
  state_tensor <- tf$convert_to_tensor(matrix(state, nrow = 1), dtype = tf$float32)
  
  # Now the model can predict using the tensor.
  q_values <- model(state_tensor)
  return(which.max(as.array(q_values)) - 1)
}

# --- Learning Function (The Training Step) ---
agent_learn <- function(q_net, target_net, batch_size) {
  
  # 1. Sample a batch of experiences from the replay buffer
  batch <- sample_batch(batch_size)
  
  # 2. Use the MAIN network to decide which action is best for the next state
  q_next_main <- q_net %>% predict_on_batch(batch$S2)
  best_actions_next <- apply(q_next_main, 1, which.max)
  
  # 3. Use the TARGET network to get the value of those best actions
  q_next_target <- target_net %>% predict_on_batch(batch$S2)
  
  # We select the Q-values from the target network using the actions chosen by the main network
  q_max <- q_next_target[cbind(seq_len(batch_size), best_actions_next)]
  
  # 4. Calculate the target Q-value (y). This is the same as before.
  # y = reward + gamma * q_max (unless it's the last step)
  y <- batch$R + (1 - batch$D) * gamma * q_max
  
  # 5. Prepare the training target. Get the current predictions...
  y_keras <- q_net %>% predict_on_batch(batch$S)
  
  # ...and update the value for the action that was actually taken.
  # Assuming your actions are 0, 1, 2, we use batch$A + 1 for R's 1-based indexing.
  y_keras[cbind(seq_len(batch_size), batch$A + 1)] <- y
  
  # 6. Train the main Q-network on the updated targets
  q_net %>% train_on_batch(batch$S, y_keras)
}