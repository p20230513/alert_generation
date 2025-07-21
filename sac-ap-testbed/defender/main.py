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

def initialize_encoder():
    
    print("Initializing OneHotEncoder with predefined alert categories...")

    # Create a representative, in-memory dataset with examples of categories
    # we expect to see from our Scapy traffic and general ET Open rules.
    known_categories = {
        'proto': [
            'TCP', 'UDP', 'ICMP'
            ],
        'alert.category': [
            'Misc activity', 
            'Potentially Bad Traffic', 
            'Attempted Information Leak',
            'A Network Trojan was detected',
            'Attempted-Dos', # Likely from ICMP/SYN floods
            'Denial of Service', # Likely from ICMP/SYN floods
            'Generic Protocol Command Decode'
            ]
    }
    
    # To make the DataFrame for the encoder, we need all combinations.
    # A simpler way is to ensure all categories are present. We can create
    # a list of dictionaries.
    fit_data = []
    max_len = max(len(v) for v in known_categories.values())
    for i in range(max_len):
        row = {}
        for key, values in known_categories.items():
            row[key] = values[i % len(values)] # Cycle through values
        fit_data.append(row)

    df_fit = pd.DataFrame(fit_data)

    # Instantiate the encoder. 'handle_unknown' is crucial for dealing with
    # new, unseen alert categories in live traffic without crashing.
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Fit the encoder to our known categories
    encoder.fit(df_fit[CATEGORICAL_FEATURES])

    # Dynamically calculate the state dimension based on the encoder's output features
    # This is the most robust way to define the state size.
    num_encoded_features = len(encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    state_dim = MAX_ALERTS_IN_STATE * num_encoded_features
    
    print(f"Encoder initialized. State dimension is set to: {state_dim}")

    # Return the fitted encoder, the feature columns it expects, and the calculated state dimension
    return encoder, CATEGORICAL_FEATURES, state_dim

def main():
    """Main simulation loop with dynamic Scapy traffic generation."""
    client = docker.from_env()
    replay_container = client.containers.get(REPLAY_CONTAINER_NAME)
    ids_ip = "172.20.0.10" # The static IP of our IDS container

    global STATE_DIM
    encoder, feature_cols, STATE_DIM = initialize_encoder()

    agent = SacApAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, action_bound=ACTION_BOUND)
    
    last_log_position = 0
    
    # Define the scenarios to cycle through during training
    scenarios = ["benign", "syn_flood", "icmp_flood", "benign"] # More benign traffic

    for episode in range(1, 101):
        print(f"\n--- Starting Episode {episode} ---")
        
        # 1. Choose a traffic scenario for this episode
        current_scenario = scenarios[episode % len(scenarios)]
        print(f"Selected scenario: {current_scenario}")

        # 2. Execute the Scapy script in the traffic-replay container
        traffic_command = (
            f"python /app/traffic_generator.py "
            f"--target-ip {ids_ip} "
            f"--scenario {current_scenario} "
            f"--duration 15"
        )
        print("Starting traffic generation...")
        replay_container.exec_run(traffic_command)
        
        # Give Suricata time to process the aftermath
        time.sleep(5)
        print("Traffic generation finished. Processing alerts...")

        # 3. Read and process alerts (this part remains the same)
        new_alerts_raw = []
        try:
            with open(SURICATA_LOG_FILE, 'r') as f:
                f.seek(last_log_position)
                lines = f.readlines()
                if lines:
                    for line in lines:
                        new_alerts_raw.append(json.loads(line))
                    last_log_position = f.tell()
        except FileNotFoundError:
            print("Alert log file not found. Waiting...")
            time.sleep(5)
            continue
        
        if not new_alerts_raw:
            print("No new alerts generated in this episode.")
            continue
            
        # ... (The rest of the alert processing, agent action, reward, and training loop is identical to the previous version) ...
        alerts_df = pd.json_normalize(new_alerts_raw)
        alerts_df = alerts_df[alerts_df['event_type'] == 'alert'].reset_index(drop=True)

        if alerts_df.empty:
            print("No valid alerts to process.")
            continue
        
        current_state, original_alerts_df, _ = process_state(alerts_df, encoder, feature_cols)
        action_priorities, _ = agent.get_action(current_state)
        investigation_indices = np.argsort(action_priorities)[-DEFENDER_BUDGET:]
        reward = get_reward(investigation_indices, original_alerts_df)
        next_state = np.zeros(STATE_DIM)
        done = True

        agent.replay_buffer.add(current_state, action_priorities, reward, next_state, done)
        agent.train(batch_size=64)
        
        print(f"Episode {episode} Training Complete. Reward: {reward}")


    # Save the final model
    print("\nTraining finished. Saving model weights...")
    agent.actor.save_weights("sac_actor.h5")
    print("Model weights saved to sac_actor.h5")

if __name__ == "__main__":
    main()
