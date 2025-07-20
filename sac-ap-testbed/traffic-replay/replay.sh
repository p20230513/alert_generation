#!/bin/bash

# Usage: ./replay.sh <pcap_file_in_data_dir> <replay_speed_in_Mbps>

PCAP_FILE="/data/$1"
REPLAY_SPEED="$2"
TARGET_IP="172.20.0.10" # Static IP of the IDS container

if [ -z "$PCAP_FILE" ] || [ -z "$REPLAY_SPEED" ]; then
  echo "Usage: $0 <pcap_filename> <replay_speed_in_Mbps>"
  exit 1
fi

echo "Replaying $PCAP_FILE to $TARGET_IP at $REPLAY_SPEED Mbps..."

# The interface tcpreplay should use is found automatically.
# We need to find the interface that is on the same subnet as our target.
INTERFACE=$(ip -o -4 addr show | awk -v ip="$TARGET_IP" '$0 ~ "172.20" {print $2}')

# Use tcpreplay to send the traffic
tcpreplay --intf1=$INTERFACE --mbps=$REPLAY_SPEED $PCAP_FILE

echo "Replay finished."