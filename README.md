# FedlEarn

## Overview  
FedlEarn is a Python-based implementation of a federated learning workflow. Clients train local models on their data, the server aggregates parameters, and finally the global model can be reconstructed from the aggregated weights.

## Features  
- Multiple client scripts (`client.py`) that train on separate datasets (e.g., `client11.csv`, `client21.csv`, …)  
- A server script (`server.py`) that collects model updates from clients and performs parameter aggregation.  
- Model checkpoint files (`model_upto1.npz`, `model_upto2.npz`, `model_upto3.npz`) that store the evolving global model parameters.  
- Reconstruction of the global model via `remodel2.py` (and similarly `remodel1.py`, `remodel3.py`).  
- A simple UI script (`ui3.py`) for interacting with or visualizing results.  
- A `run.sh` shell script to orchestrate the workflow end-to-end.

## Repository Structure  
