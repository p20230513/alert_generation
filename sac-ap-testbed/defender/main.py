import time
import docker
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from agent import SacApAgent

# --- Configuration ---
REPLAY_CONTAINER_NAME = "traffic-replay"
SURICATA_LOG_FILE = "/var/log/suricata/eve.json"
PCAP_FILE_TO_REPLAY = "Monday-WorkingHours.pcap"
REPLAY_SPEED_MBPS = 20
MAX_ALERTS_IN_STATE = 50  # Fixed number of alerts to consider in a state
DEFENDER_BUDGET = 5      # How many alerts to investigate per step

# --- State and Action Dimensions ---
# These must be defined based on your feature engineering
CATEGORICAL_FEATURES = ['proto', 'alert.category']
# The state dimension will be MAX_ALERTS * num_features after one-hot encoding
# We will calculate this dynamically later.
ACTION_DIM = MAX_ALERTS_IN_STATE # Action is a priority score for each alert
ACTION_BOUND = 1.0 # Actions are between -1 and 1

def process_state(alerts_df, one_hot_encoder, feature_cols):
    """Convert a dataframe of alerts into a fixed-size numerical state vector."""
    if alerts_df.empty:
        # Return a zero vector if no alerts
        return np.zeros(STATE_DIM), None, None

    # Truncate or pad to ensure fixed size
    if len(alerts_df) > MAX_ALERTS_IN_STATE:
        alerts_df = alerts_df.head(MAX_ALERTS_IN_STATE)
    
    original_alerts = alerts_df.copy() # Keep for reward calculation
    
    # One-Hot Encode categorical features
    encoded_cats = one_hot_encoder.transform(alerts_df[CATEGORICAL_FEATURES]).toarray()
    
    # For simplicity, we ignore other features for now.
    # In a real setup, you would normalize numerical features and concatenate.
    state = encoded_cats

    # Pad with zeros if there are fewer alerts than MAX_ALERTS
    padding_rows = MAX_ALERTS_IN_STATE - len(state)
    if padding_rows > 0:
        padding = np.zeros((padding_rows, state.shape[1]))
        state = np.vstack([state, padding])
    
    # Flatten to create a single state vector
    return state.flatten(), original_alerts, one_hot_encoder

def get_reward(investigation_indices, alerts_df):
    """
    Simulate a reward function based on a 'ground truth'.
    This is a placeholder and requires a real mapping for accuracy.
    """
    if alerts_df is None or alerts_df.empty:
        return 0

    reward = 0
    # Simulate ground truth: if signature contains 'attack', it's a true positive
    is_attack = alerts_df['alert.signature'].str.contains('attack|malware|exploit', case=False, na=False)
    
    # Reward for investigating true positives
    for idx in investigation_indices:
        if idx < len(is_attack) and is_attack.iloc[idx]:
            reward += 10 # High reward for catching an attack
        else:
            reward -= 1  # Small penalty for investigating benign alerts

    # Penalty for missing attacks
    missed_attacks = is_attack.sum() - sum(is_attack.iloc[idx] for idx in investigation_indices if idx < len(is_attack))
    reward -= missed_attacks * 10

    return reward

def initialize_encoder(pcap_file):
    """
    Pre-scans a sample of alerts to initialize the OneHotEncoder, so it knows all possible categories.
    This is a practical trick to handle unseen categories during runtime.
    """
    print("Initializing OneHotEncoder by pre-scanning for alert categories...")
    # This is a simplified version. A real one would run Suricata on a sample PCAP.
    # For now, we'll use a predefined set of common categories.
    known_categories = {
        'proto': ['TCP', 'UDP', 'ICMP'],
        'alert.category': ['A Network Trojan was detected', 'Misc activity', 'Potentially Bad Traffic', 'Attempted User Privilege Gain']
    }
    df = pd.DataFrame(known_categories)
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(df)
    
    # Calculate state dimension based on the encoder
    state_dim = MAX_ALERTS_IN_STATE * len(encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    print(f"State dimension calculated: {state_dim}")
    
    return encoder, encoder.get_feature_names_out(CATEGORICAL_FEATURES), state_dim

def main():
    """Main simulation loop."""
    client = docker.from_env()
    replay_container = client.containers.get(REPLAY_CONTAINER_NAME)

    # Initialize encoder and get state dimension
    global STATE_DIM
    encoder, feature_cols, STATE_DIM = initialize_encoder(PCAP_FILE_TO_REPLAY)

    # Initialize the agent
    agent = SacApAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, action_bound=ACTION_BOUND)
    
    last_log_position = 0

    for episode in range(1, 101):
        print(f"\n--- Starting Episode {episode} ---")
        
        # Trigger traffic replay
        replay_command = f"/app/replay.sh {PCAP_FILE_TO_REPLAY} {REPLAY_SPEED_MBPS}"
        replay_container.exec_run(replay_command)
        print(f"Replay of '{PCAP_FILE_TO_REPLAY}' started. Waiting for alerts...")
        time.sleep(20) # Give Suricata time to process

        # Read new alerts
        new_alerts_raw = []
        try:
            with open(SURICATA_LOG_FILE, 'r') as f:
                f.seek(last_log_position)
                lines = f.readlines()
                for line in lines:
                    new_alerts_raw.append(json.loads(line))
                last_log_position = f.tell()
        except FileNotFoundError:
            print("Alert log file not found. Waiting...")
            time.sleep(5)
            continue
        
        if not new_alerts_raw:
            print("No new alerts.")
            continue

        alerts_df = pd.json_normalize(new_alerts_raw)
        alerts_df = alerts_df[alerts_df['event_type'] == 'alert']

        if alerts_df.empty:
            print("No valid alerts to process.")
            continue
        
        # Get current state
        current_state, original_alerts_df, _ = process_state(alerts_df, encoder, feature_cols)
        
        # Get action from agent
        action_priorities, _ = agent.get_action(current_state)
        
        # Defender investigates the top N alerts based on priority
        # The action gives priorities; we select the highest ones
        investigation_indices = np.argsort(action_priorities)[-DEFENDER_BUDGET:]
        
        # Get reward
        reward = get_reward(investigation_indices, original_alerts_df)
        
        # For simplicity, next_state is a zero vector as each replay is one 'step'
        next_state = np.zeros(STATE_DIM)
        done = True

        # Store experience and train
        agent.replay_buffer.add(current_state, action_priorities, reward, next_state, done)
        agent.train(batch_size=64)
        
        print(f"Episode {episode} finished. Reward: {reward}")

if __name__ == "__main__":
    main()