# =================================================================
# FINAL EVALUATION & VISUALIZATION SCRIPT
# =================================================================

## 1. SETUP: LOAD LIBRARIES AND ALL SCRIPTS
# -----------------------------------------------------------------
library(tidyverse)
library(lubridate)
library(gridExtra) # For arranging plots

source("config.R")         # Hyper-parameters from Person C
source("data_loader.R")   # Data loading from Person C
source("agent_logic.R")     # Agent functions from Person B
source("battery_settings.R")


## 2. RUN ALL EVALUATIONS ON TEST DATA
# -----------------------------------------------------------------
cat("\n--- Evaluating Benchmark Policies ---\n")
# Random Policy
random_env <- create_env(price_test)
random_results <- evaluate_policy(random_policy, price_test, random_env)
cat(sprintf("Final Random Policy Profit: €%.2f\n", random_results$final_profit))

# Rule-Based Policy
rule_env <- create_env(price_test)
rule_results <- evaluate_policy(rule_based_policy, price_test, rule_env)
cat(sprintf("Final Rule-Based Policy Profit: €%.2f\n", rule_results$final_profit))


cat("\n--- Evaluating Reinforcement Learning Agents ---\n")
# Build the empty network structure once
q_network_shell <- build_q_network(state_size = state_size, action_size = action_size)
prices_norm_test <- (price_test - price_mean) / price_sd

# --- Evaluate T1 (Baseline DQN) Agent ---
load_model_weights_hdf5(q_network_shell, "dqn_t1_battery_model.h5") # Use T1 weights
t1_results <- evaluate_dqn_agent(q_network_shell, price_test, prices_norm_test)
cat(sprintf("Final T1 DQN Profit: €%.2f\n", t1_results$final_profit))

# --- Evaluate T4 (Enhanced DDQN) Agent ---
load_model_weights_hdf5(q_network_shell, "dqn_t4_battery_model.h5") # Use T4 weights
t4_results <- evaluate_dqn_agent(q_network_shell, price_test, prices_norm_test)
cat(sprintf("Final T4 DDQN Profit: €%.2f\n", t4_results$final_profit))


## 3. PREPARE DATA FOR PLOTTING
# -----------------------------------------------------------------
# Choose a specific, interesting period for visualization

eval_period <- 1:(96 * 7) # First 7 days

# Create a clean plotting function to avoid repeating code
create_plot_data <- function(results, test_data, period) {
  tibble(
    time = test_data$datetime_cet[period],
    price = test_data$price[period],
    soc = results$soc_history[period],
    action = factor(
      results$action_history[period], 
      levels = 0:2, 
      labels = c("Hold", "Charge", "Discharge")
    )
  )
}

# Create the dataframes for plotting
plot_data_rule_based <- create_plot_data(rule_results, test_data, eval_period)
plot_data_t1 <- create_plot_data(t1_results, test_data, eval_period)
plot_data_t4 <- create_plot_data(t4_results, test_data, eval_period)


## 4. GENERATE CLEANER PLOTS
# -----------------------------------------------------------------
# This is our master plotting function to make plots look nice and consistent

generate_comparison_plot <- function(plot_data, agent_title) {
  
  # Determine a good scaling factor for SoC to make it visible
  soc_scaler <- max(plot_data$price, na.rm = TRUE) * 0.9 
  
  # Create data for the action band rectangles
  action_data <- plot_data %>%
    filter(action != "Hold") %>%
    # Define the vertical position and height of the band
    mutate(ymin = -max(plot_data$price, na.rm = TRUE) * 0.15, 
           ymax = -max(plot_data$price, na.rm = TRUE) * 0.05) 
  
  ggplot(plot_data, aes(x = time)) +
    
    # --- Layers ---
    # 1. Action Band at the bottom using geom_rect
    geom_rect(data = action_data, 
              aes(xmin = time - minutes(7), xmax = time + minutes(7), 
                  ymin = ymin, ymax = ymax, fill = action), alpha = 0.9) +
    
    # 2. Main plot lines
    geom_line(aes(y = price, color = "Price"), size = 0.7) +
    geom_line(aes(y = soc * soc_scaler, color = "SoC"), linetype = "dashed", size = 1) +
    
    # Add a horizontal line at y=0 for reference
    geom_hline(yintercept = 0, linetype = "dotted", color = "gray50") +
    
    # --- Aesthetics and Labels ---
    scale_color_manual(name = "Metric", values = c("Price" = "black", "SoC" = "royalblue")) +
    scale_fill_manual(name = "Action", values = c("Charge" = "mediumseagreen", "Discharge" = "firebrick")) +
    
    scale_y_continuous(
      name = "Price (€/MWh)",
      sec.axis = sec_axis(~./soc_scaler, name = "State of Charge", labels = scales::percent)
    ) +
    labs(
      title = agent_title,
      subtitle = "Agent strategy shown with price, SoC, and a dedicated action band",
      x = NULL
    ) +
    guides(color = guide_legend(order = 1), fill = guide_legend(order = 2)) +
    theme_light(base_size = 14) +
    theme(legend.position = "bottom",
          plot.title = element_text(face = "bold", size = 16),
          plot.subtitle = element_text(size = 11, color = "gray30"))
}


# --- Create the plots ---
p_rule_based <- generate_comparison_plot(plot_data_rule_based, "Rule-Based Agent Strategy")
p_t1_agent <- generate_comparison_plot(plot_data_t1, "Baseline DQN (T1) Agent Strategy")
p_t4_agent <- generate_comparison_plot(plot_data_t4, "Enhanced DDQN (T4) Agent Strategy")

# --- Display or Save the plots ---

# Display all three plots arranged vertically
grid.arrange(p_rule_based, p_t1_agent, p_t4_agent, ncol = 1)

# To save them for your report:
ggsave("plot_rule_based.png", plot = p_rule_based, width = 12, height = 7)
ggsave("plot_t1_agent.png", plot = p_t1_agent, width = 12, height = 7)
ggsave("plot_t4_agent.png", plot = p_t4_agent, width = 12, height = 7)

cat("\nAll plots generated and saved.\n")

# --- Small comparison of DQN Vs. DDQN ---
library(ggplot2)
library(tidyverse)

plot_period <- 1:70176  # or your chosen range

comparison_df <- bind_rows(
  tibble(
    time_index = plot_period,
    time = test_data$datetime_cet[plot_period],
    price = price_test[plot_period],
    agent = "DQN (Aggressive)",
    soc = t1_results$soc_history[plot_period],
    action = factor(t1_results$action_history[plot_period], levels = 0:2, labels = c("Hold", "Charge", "Discharge"))
  ),
  tibble(
    time_index = plot_period,
    time = test_data$datetime_cet[plot_period],
    price = price_test[plot_period],
    agent = "DDQN (Cautious)",
    soc = t4_results$soc_history[plot_period],
    action = factor(t4_results$action_history[plot_period], levels = 0:2, labels = c("Hold", "Charge", "Discharge"))
  )
)

# 1. Calculate SoC changes per step
comparison_df <- comparison_df %>%
  arrange(agent, time) %>%
  group_by(agent) %>%
  mutate(
    soc_change = soc - lag(soc)
  ) %>%
  ungroup()

# 2. Action counts per agent
action_counts <- comparison_df %>%
  group_by(agent, action) %>%
  summarise(count = n()) %>%
  arrange(agent, desc(count))

action_percent <- action_counts %>%
  group_by(agent) %>%
  mutate(
    total_actions = sum(count),
    pct = count / total_actions * 100
  ) %>%
  ungroup()

t4_comparison <- ggplot(action_percent, aes(x = action, y = pct, fill = agent)) +
  geom_col(position = "dodge") +
  labs(
    title = "Relative Frequency of Actions by Agent",
    x = "Action",
    y = "Percentage of Total Actions (%)",
    fill = "Agent"
  ) +
  scale_fill_manual(values = c("DDQN (Cautious)" = "steelblue", "DQN (Aggressive)" = "firebrick")) +
  theme_minimal()

ggsave("plot_t4_comparison.png", plot = t4_comparison)
