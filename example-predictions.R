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

# -- Current Season Final 5 Contestants -- #

# STEVEN'S DATA IS NOT UPDATED YET, HE WAS VOTED OUT ON DEC 10th
# I WILL REMOVE HIM FROM THE DATA FOR THIS EXAMPLE

current_finalists = cast %>%
  filter(season == 49,
         castaway != "Steven",
         is.na(episode))

finalist_ids = current_finalists$castaway_id

current_scores = cast_scores %>%
  filter(castaway_id %in% finalist_ids)



# --- Identify Winners for Each Season ---
winners <- cast %>%
  select(season, castaway, winner) %>%
  na.omit()

# --- Select Top 5 Contestants Per Season by Outlast Score ---
top5_outlast <- cast_scores %>%
  group_by(season) %>%
  arrange(desc(score_outlast)) %>%
  slice_head(n = 5) %>%
  select(
    season, castaway, castaway_id, score_outlast,
    n_idols_found, n_adv_found, n_tribals, n_votes_received
  ) %>%
  ungroup()

# --- Merge Top 5 Data with Winners ---
top_win_merge <- top5_outlast %>%
  left_join(winners, join_by(season == season, castaway == castaway)) %>%
  mutate(
    n_idols_found = replace_na(n_idols_found, 0),
    n_adv_found = replace_na(n_adv_found, 0),
    n_tribals = replace_na(n_tribals, 0),
    winner = as.numeric(winner)
  )

# --- Prepare Training Data (All Seasons Except Current [49]) ---
train_x <- top_win_merge %>%
  filter(season != 49) %>%
  select(n_idols_found, n_adv_found, n_tribals, n_votes_received, winner)

train_y <- train_x$winner
train_x <- train_x %>% select(-winner)

# Convert to Matrix for XGBoost
m_train_x = as.matrix(train_x)

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

# --- Prepare Current Season ---
current_season_data = current_scores %>%
  select(n_idols_found, n_adv_found, n_tribals, n_votes_received, castaway)

  

current_castaways <- current_season_data$castaway
current_season_data <- current_season_data %>% select(-castaway)

# Convert to Matrix for Prediction
m_data = as.matrix(current_season_data)

# --- Generate Predictions ---
preds <- predict(bst, m_data)

# --- Combine Predictions with Castaway Info ---
check <- data.frame(
  Name = current_castaways,
  Predictions = preds,
  Probabilities = preds / sum(preds)
)

# --- View Results ---
check

# An example of what's expected in your .csv submission.
# Your code should reproduce your predictions EXACTLY

example_submission = check %>%
  select(Name, Probabilities)

write.csv(example_submission, "example_submission.csv", row.names = FALSE)

