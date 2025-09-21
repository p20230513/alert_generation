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
NUMERICAL_FEATURES = ['alert.severity']
# The state dimension will be MAX_ALERTS * num_features after one-hot encoding
# We will calculate this dynamically later.
ACTION_DIM = MAX_ALERTS_IN_STATE # Action is a priority score for each alert
ACTION_BOUND = 1.0 # Actions are between -1 and 1

def process_state(alerts_df, one_hot_encoder, feature_cols):
    """Convert a dataframe of alerts into a fixed-size numerical state vector."""
    if alerts_df.empty:
        return np.zeros(STATE_DIM), None, None

    if len(alerts_df) > MAX_ALERTS_IN_STATE:
        alerts_df = alerts_df.head(MAX_ALERTS_IN_STATE)
    
    original_alerts = alerts_df.copy()
    
    # 1. Process Categorical Features
    encoded_cats = one_hot_encoder.transform(alerts_df[CATEGORICAL_FEATURES])
    
    # 2. Process Numerical Features (and normalize them)
    numerical_data = alerts_df[NUMERICAL_FEATURES].copy()
    # Normalize severity: divide by a reasonable max (e.g., 5) to keep it small
    numerical_data['alert.severity'] = numerical_data['alert.severity'] / 5.0
    
    # 3. Combine them
    state_parts = [encoded_cats, numerical_data.to_numpy()]
    state = np.hstack(state_parts)

    # 4. Pad with zeros if needed
    padding_rows = MAX_ALERTS_IN_STATE - len(state)
    if padding_rows > 0:
        padding = np.zeros((padding_rows, state.shape[1]))
        state = np.vstack([state, padding])
    
    return state.flatten(), original_alerts, one_hot_encoder

def get_reward(investigation_indices, alerts_df):
    """
    Calculates a more nuanced reward based on the type of alert investigated.
    """
    if alerts_df is None or alerts_df.empty:
        return 0

    reward = 0
    
    # --- NEW: Define what constitutes a true attack vs. a benign alert ---
    is_attack = alerts_df['alert.signature'].str.contains('Flood|DOS|ICMP', case=False, na=False)
    is_benign = alerts_df['alert.signature'].str.contains('Benign Web Traffic', case=False, na=False)
    
    # Loop through the alerts the agent chose to investigate
    for idx in investigation_indices:
        if idx < len(alerts_df):
            if is_attack.iloc[idx]:
                reward += 10  # High reward for catching a real attack
            elif is_benign.iloc[idx]:
                reward -= 1   # Small penalty for wasting time on a known benign alert
            else:
                # This can be for other ET Open rules that might fire
                reward -= 2   # A neutral penalty for unknown alerts

    # --- NEW: The penalty for missed attacks should ONLY apply to true attacks ---
    total_attacks = is_attack.sum()
    investigated_attacks = sum(is_attack.iloc[idx] for idx in investigation_indices if idx < len(is_attack))
    
    missed_attacks = total_attacks - investigated_attacks
    reward -= missed_attacks * 10

    return reward

def initialize_encoder():
    """Initializes the OneHotEncoder and calculates the final state dimension."""
    print("Initializing OneHotEncoder with predefined alert categories...")

    known_categories = {
        'proto': ['TCP', 'UDP', 'ICMP'],
        'alert.category': ['Misc activity', 'Potentially Bad Traffic', 'Attempted Information Leak',
                           'A Network Trojan was detected', 'Attempted-Dos', 'Denial of Service',
                           'Generic Protocol Command Decode']
    }
    fit_data = []
    max_len = max(len(v) for v in known_categories.values())
    for i in range(max_len):
        row = {}
        for key, values in known_categories.items():
            row[key] = values[i % len(values)]
        fit_data.append(row)
    df_fit = pd.DataFrame(fit_data)

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(df_fit[CATEGORICAL_FEATURES])

    # NEW LOGIC: State dimension is now encoded + numerical features
    num_encoded_features = len(encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    num_numerical_features = len(NUMERICAL_FEATURES)
    state_dim = MAX_ALERTS_IN_STATE * (num_encoded_features + num_numerical_features)
    
    print(f"Encoder initialized. State dimension is set to: {state_dim}")
    return encoder, CATEGORICAL_FEATURES, state_dim

def main():
    """Main simulation loop with dynamic Scapy traffic and robust polling."""
    client = docker.from_env()
    replay_container = client.containers.get(REPLAY_CONTAINER_NAME)
    ids_ip = "172.20.0.10"

    # Action 2 Change: Make sure you have the updated initialize_encoder
    global STATE_DIM
    encoder, feature_cols, STATE_DIM = initialize_encoder()

    agent = SacApAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, action_bound=ACTION_BOUND)
    
    # We will now manage the log position inside the loop for reliability
    last_log_position = 0
    
    scenarios = ["benign", "syn_flood", "icmp_flood", "benign"]

    for episode in range(1, 1001): # Let's plan to run for more episodes
        print(f"\n--- Starting Episode {episode} ---")
        
        current_scenario = scenarios[episode % len(scenarios)]
        print(f"Selected scenario: {current_scenario}")

        traffic_command = (f"python /app/traffic_generator.py --target-ip {ids_ip} "
                         f"--scenario {current_scenario} --duration 15")
        
        print("Starting traffic generation...")
        replay_container.exec_run(traffic_command)
        print("Traffic generation finished. Waiting for IDS to process and write logs...")

        # --- NEW ROBUST POLLING LOGIC ---
        new_alerts_raw = []
        alerts_found = False
        max_wait_time = 20  # Wait for a maximum of 20 seconds
        start_time = time.time()
        
        # Keep checking for new log entries until the timeout
        while time.time() - start_time < max_wait_time:
            try:
                with open(SURICATA_LOG_FILE, 'r') as f:
                    f.seek(last_log_position)
                    lines = f.readlines()
                    if lines:
                        print(f"Found {len(lines)} new log entries. Processing...")
                        for line in lines:
                            new_alerts_raw.append(json.loads(line))
                        last_log_position = f.tell()
                        alerts_found = True
                        break # Exit the loop once alerts are found
            except FileNotFoundError:
                # The log file might not exist yet, which is fine
                pass
            except json.JSONDecodeError:
                print("Warning: JSON decode error while reading logs. Retrying...")
            
            time.sleep(1) # Wait 1 second before checking again

        if not alerts_found:
            print("No new alerts found within the timeout period.")
            continue
        # --- END OF POLLING LOGIC ---

        alerts_df = pd.json_normalize(new_alerts_raw)
        alerts_df = alerts_df[alerts_df['event_type'] == 'alert'].reset_index(drop=True)

        if alerts_df.empty:
            print("Log entries were found, but none were valid alerts.")
            continue
        
        # Action 2 Change: Ensure process_state is the updated version
        current_state, original_alerts_df, _ = process_state(alerts_df, encoder, feature_cols)
        
        # The rest of the loop remains the same
        action_priorities, _ = agent.get_action(current_state)
        investigation_indices = np.argsort(action_priorities)[-DEFENDER_BUDGET:]
        reward = get_reward(investigation_indices, original_alerts_df)
        next_state = np.zeros(STATE_DIM)
        done = True

        agent.replay_buffer.add(current_state, action_priorities, reward, next_state, done)
        agent.train(batch_size=64)
        
        print(f"Episode {episode} Training Complete. Reward: {reward}")

    print("\nTraining finished. Saving model weights...")
    agent.actor.save_weights("sac_actor.h5")
    print("Model weights saved to sac_actor.h5")

if __name__ == "__main__":
    main()
