# LSTM parameters
INPUT_SIZE = 24  # 24 hours per day.
OUTPUT_SIZE = 24  # 24 hours per day.
HIDDEN_SIZE = 6  # Dimension of hidden state of LSTM.

# Training parameters.
EPOCHS = 20
BATCH_SIZE = 64
TRAIN_RATIO = 0.8
VALI_RATIO = 0.1
TEST_RATIO = 1 - TRAIN_RATIO - VALI_RATIO

# Data related.
LOOK_AHEAD_DAY = 7
HOUR_PER_DAY = 24
