# ============================
# Shared Parameters
# ============================

# -- Battery Physical & Economic Constraints (from Task Sheet) --
CAPACITY_MWH             <- 1.0     # MWh, usable energy
POWER_LIMIT_MWH          <- 0.5     # MW, max charge/discharge rate hourly
POWER_LIMIT_MW15         <- 0.5 / 4 # MW, max charge/discharge rate hourly
EFFICIENCY               <- 0.90    # Round-trip efficiency (90%)
DEGRADATION_COST_EUR_MWH <- 20      # EUR per MWh of throughput
SOC_MIN                 <- 0.0    # state of charge min
SOC_MAX                 <- 1.0    # state of charge max

# -- RL Agent & Network Hyper-parameters --
window_size        <- 24 * 4              # Look-back window: 24 hours (96 steps)
state_size         <- window_size + 1     # State = prices + current SOC
action_size        <- 3                   # 0 = Hold, 1 = Charge, 2 = Discharge

hidden_units       <- 128
learning_rate      <- 0.001
gamma              <- 0.99                # Discount factor

episodes           <- 50                  # Each = one full pass through the training data
batch_size         <- 128
replay_capacity    <- 10000               # Buffer to store experiences
warmup_mem         <- 1000                # Agent won't train until buffer has this many memories
target_sync_freq   <- 10                  # Sync

# -- Exploration Strategy --
eps_start          <- 1.0   # Starting exploration rate
eps_final          <- 0.05  # Final exploration rate
