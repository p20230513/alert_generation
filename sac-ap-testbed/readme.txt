## Directory structure:

sac-ap-testbed/
│
├── docker-compose.yml
├── data/ (Contains your .pcap files)
├── logs/suricata/
├── ids-config/suricata.yaml
└── traffic-replay/
    ├── Dockerfile             
    ├── requirements.txt        
    └── traffic_generator.py 
│
└── defender/
├── Dockerfile
├── requirements.txt
├── main.py
└── agent.py

1. Open a terminal in the sac-ap-testbed root directory.

2. Start all the services in detached mode:
#docker-compose up -d --build

3. The defender container will automatically start the simulation. To watch the output and see the simulation progress, run:
#docker-compose logs -f defender
