# EXAMPLE OF HOW YOU CAN MAKE SURVIVOR WINNER PREDICTIONS
# USING PAST SEASONS AS TRAINING DATA. THIS IS ONLY A BASIC
# MODEL. YOU CAN ADD VARIABLES, ENGINEER DIFFERENT VARIABLES
# SYNTHESIZE MORE TRANING DATA. DO WHATEVER YOU NEED.

# --- Load Required Libraries ---
library(survivoR)
library(xgboost)
library(dplyr)
library(tidyr)

# --- Load Survivor Data ---
cast_det <- survivoR::castaway_details
cast <- survivoR::castaways
chal <- survivoR::challenge_results
cast_scores <- survivoR::castaway_scores
vh <- survivoR::vote_history

# --- Identify Winners for Each Season ---
winners <- cast %>%
  select(season, castaway, winner) %>%
  na.omit()

# --- Select Top 4 Contestants Per Season by Outlast Score ---
top4_outlast <- cast_scores %>%
  group_by(season) %>%
  arrange(desc(score_outlast)) %>%
  slice_head(n = 4) %>%
  select(
    season, castaway, castaway_id, score_outlast,
    n_idols_found, n_adv_found, n_tribals, n_votes_received
  ) %>%
  ungroup()

# --- Merge Top 4 Data with Winners ---
top_win_merge <- top4_outlast %>%
  left_join(winners, join_by(season == season, castaway == castaway)) %>%
  mutate(
    n_idols_found = replace_na(n_idols_found, 0),
    n_adv_found = replace_na(n_adv_found, 0),
    n_tribals = replace_na(n_tribals, 0),
    winner = as.numeric(winner)
  )

# --- Prepare Training Data (All Seasons Except 48) ---
train_x <- top_win_merge %>%
  filter(season != 48) %>%
  select(n_idols_found, n_adv_found, n_tribals, n_votes_received, winner)

train_y <- train_x$winner
train_x <- train_x %>% select(-winner)

# Convert to Matrix for XGBoost
m_train_x <- model.matrix(~ . - 1, data = train_x)  # Remove intercept term

set.seed(123)
# --- Train XGBoost Model ---
bst <- xgboost(
  data = m_train_x,
  label = train_y,
  nrounds = 200,
  objective = "binary:logistic",
  max_depth = 3,
  verbose = 0
)

# --- Prepare Test Data (Season 48) ---
test_x <- top_win_merge %>%
  filter(season == 48) %>%
  select(n_idols_found, n_adv_found, n_tribals, n_votes_received, winner, castaway)

test_castaway <- test_x$castaway
test_y <- test_x$winner
test_x <- test_x %>% select(-winner, -castaway)

# Convert to Matrix for Prediction
m_test_x <- model.matrix(~ . - 1, data = test_x)

# --- Generate Predictions ---
preds <- predict(bst, m_test_x)

# --- Combine Predictions with Castaway Info ---
check <- data.frame(
  Name = test_castaway,
  Predictions = preds,
  Probabilities = preds / sum(preds),
  Actual = test_y
)

# --- View Results ---
check

# An example of what's expected in your .csv submission.
# Your code should reproduce your predictions EXACTLY

example_submission = check %>%
  select(Name, Probabilities)


