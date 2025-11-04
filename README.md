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
FedlEarn/
├── Dataset/ # client datasets and combined dataset

│ ├── client11.csv

│ ├── client21.csv

│ ├── …

│ ├── combined1.csv # merged dataset from all clients

│ └── test.csv # test dataset

├── client.py # client side training code

├── server.py # aggregation/ server code

├── remodel2.py # reconstruct model from checkpoint model_upto2.npz

├── run.sh # shell script to run full pipeline

├── ui3.py # UI/visualization script

└── README.md # this file

---

## Getting Started 🚀
### Prerequisites
* **Python 3.x**
* **Required Python packages** (you may install them via `pip install …`).
* Ensure all client CSV files are available in the `Dataset/` folder as listed (e.g., `client11.csv`, `client21.csv`, …)
* The dataset `combined1.csv` should be generated (or already present) containing merged data from all clients.
* A test dataset `test.csv` should be present.

### Installation & Usage
1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/Shreyas277/FedlEarn.git](https://github.com/Shreyas277/FedlEarn.git)
    cd FedlEarn
    ```
2.  **(Optional) Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate # on Unix/macOS
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt # (if you add one)
    ```
4.  **Run the workflow**:
    ```bash
    ./run.sh
    ```

> This script will:
> * Trigger client side training for each dataset
> * Aggregate the results on the server side
> * Save model checkpoint files (`model_upto1.npz`, etc)
> * Reconstruct the global model from checkpoints via `remodel*.py`
> * Optionally launch the UI (`ui3.py`) for visualization.

---

## Configuration ⚙️
You may edit `client.py` & `server.py` to change model hyperparameters, number of clients, dataset paths, or aggregation logic.

The checkpoint files are named `model_uptoX.npz` where `X` indicates up to which client’s updates were included. Adjust naming/logic as needed.

---

## How It Works (Conceptually) 🧠
* **Client Side**: Each client loads its local data CSV (e.g., `client11.csv`) and trains a local model. It then exports its model weights (e.g., via numpy) and sends them to the server.
* **Server Side**: The server collects the weights from all clients, performs an **aggregation step** (e.g., averaging) to form a new global model weight set. This is saved as `model_upto1.npz`, `model_upto2.npz`, etc.
* **Reconstruction / Deployment**: Scripts such as `remodel2.py` load the checkpoint file and build an ANN model that uses those weights for inference or further training.
* **Visualization / UI**: The `ui3.py` script offers a simple interface (CLI or GUI) to inspect results, maybe show accuracy on `test.csv`, visualize model behaviour, etc.

---
