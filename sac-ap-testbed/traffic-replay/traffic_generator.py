import argparse
import time
import random
from scapy.all import send, IP, TCP, ICMP

def generate_benign_traffic(target_ip, duration):
    """Generates random, legitimate-looking TCP traffic to common ports."""
    print(f"Generating benign traffic to {target_ip} for {duration} seconds...")
    end_time = time.time() + duration
    common_ports = [80, 443, 22, 53]
    
    while time.time() < end_time:
        src_ip = ".".join(map(str, (random.randint(1, 254) for _ in range(4))))
        dest_port = random.choice(common_ports)
        
        # Send a SYN packet, mimicking the start of a connection
        packet = IP(src=src_ip, dst=target_ip) / TCP(dport=dest_port, flags="S")
        send(packet, verbose=0)
        time.sleep(random.uniform(0.1, 0.5))

def generate_syn_flood(target_ip, duration):
    """Generates a high volume of TCP SYN packets (SYN Flood)."""
    print(f"Generating SYN FLOOD to {target_ip} for {duration} seconds...")
    end_time = time.time() + duration
    
    while time.time() < end_time:
        src_ip = ".".join(map(str, (random.randint(1, 254) for _ in range(4))))
        # Target a specific port, as is common in a DoS attack
        packet = IP(src=src_ip, dst=target_ip) / TCP(dport=80, flags="S")
        send(packet, count=10, verbose=0) # Send packets in small bursts
        time.sleep(0.05)

def generate_icmp_flood(target_ip, duration):
    """Generates a high volume of ICMP Echo Request packets (Ping Flood)."""
    print(f"Generating ICMP FLOOD to {target_ip} for {duration} seconds...")
    end_time = time.time() + duration

    while time.time() < end_time:
        src_ip = ".".join(map(str, (random.randint(1, 254) for _ in range(4))))
        packet = IP(src=src_ip, dst=target_ip) / ICMP()
        send(packet, count=10, verbose=0)
        time.sleep(0.05)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scapy Traffic Generator")
    parser.add_argument("--target-ip", required=True, help="The IP address of the target (IDS)")
    parser.add_argument("--duration", type=int, default=15, help="Duration to generate traffic in seconds")
    parser.add_argument(
        "--scenario", 
        required=True, 
        choices=["benign", "syn_flood", "icmp_flood"],
        help="The traffic scenario to generate."
    )
    
    args = parser.parse_args()
    
    if args.scenario == "benign":
        generate_benign_traffic(args.target_ip, args.duration)
    elif args.scenario == "syn_flood":
        generate_syn_flood(args.target_ip, args.duration)
    elif args.scenario == "icmp_flood":
        generate_icmp_flood(args.target_ip, args.duration)

    print(f"Finished generating scenario: {args.scenario}")
