class Config:
    # Environment settings
    TRAIN_GRID_WIDTH = 4
    TRAIN_GRID_HEIGHT = 4
    MAX_STEPS = 50

    # Action settings
    ACTIONS = 5  # 0:left, 1:right, 2:forward, 3:pickup, 4:open door

    # Q-learning Oracle
    QLEARNING_EPISODES = 3000
    QLEARNING_CONTINUE_EPISODES = 50
    EPSILON_DECAY = 0.995
    MIN_EPSILON = 0.01
    ALPHA = 0.1
    GAMMA = 0.99

    # Dataset collection
    DATASET_SIZE = 21500
    RANDOM_ACTION_RATIO = 0.0  # Force only oracle actions
    VAL_SPLIT = 0.1

    # LLM Settings
    LLM_MODEL = "bert-base-uncased"
    DROPOUT_PROB = 0.1
    MC_SAMPLES = 8
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-5
    EPOCHS = 5
    MAX_SEQ_LENGTH = 512
    TARGET_ACCURACY = 0.90

    # PPO Settings (Policy Network)
    PPO_EPSILON = 0.1
    PPO_ENTROPY_COEF = 0.001
    PPO_VALUE_COEF = 0.5
    PPO_LR = 1e-4

    # Rewards
    KEY_REWARD = 0.5
    DOOR_REWARD = 0.5
    GOAL_REWARD = 0.2
    INVALID_ACTION_PENALTY = -0.02

    # Curriculum
    CURRICULUM = {
        0: (4, 4),
        500: (5, 5),
        1000: (6, 6)
    }

    # Logging and Save Paths
    SAVE_DIR = "calibrated_llm_rl/4x4_results_qlearning"
    CHECKPOINT_EVERY = 500
    TARGET_FINE_TUNE_ACCURACY = 0.95

    EVAL_GRID_WIDTH = 4
    EVAL_GRID_HEIGHT = 4