# Federated Learning with NVFlare for Phyto-Plankton Classification

This folder contains configuration files and setup instructions for training a **phyto-plankton classification** model using **Federated Learning** with [NVFLARE](https://github.com/NVIDIA/NVFlare).

We provide setups to train the model using both the **FedAvg** and **Swarm Learning** algorithms in a federated setting.

---

# Structure of this project
```bash
├── jobs                            # Contains all federated learning job definitions for NVFLARE
│   ├── FedAvg                      # Job configuration for training using the FedAvg algorithm
│   │   ├── app                     # Application logic and components for FedAvg
│   │   │   ├── config              # Configuration files for FL clients and server
│   │   │   │   ├── config_fed_client.conf   # FedAvg client-side config (e.g., components, task settings)
│   │   │   │   └── config_fed_server.conf   # FedAvg server-side config (e.g., workflows, aggregators)
│   │   │   └── custom              # Custom Python modules used in training
│   │   │       ├── mlflow_receiver.py       # Logs training metrics to MLFlow via NVFLARE interface
│   │   │       ├── model_persistor.py       # Handles model saving/loading between training rounds
│   │   │       ├── split_data.py            # Splits dataset for each federated client
│   │   │       └── train_runfile_nvflare.py # Main training script executed by NVFLARE on each client
│   │   └── meta.conf              # Metadata file defining job roles, names, and dependencies
│   ├── split_data.py              # Optional script to globally pre-split datasets before job execution
│   └── Swarm                      # Job configuration for training using the Swarm Learning algorithm
│       ├── app                    # Application logic and components for Swarm
│       │   ├── config             # Configuration files for Swarm FL clients and server
│       │   │   ├── config_fed_client.conf   # Swarm client-side config
│       │   │   └── config_fed_server.conf   # Swarm server-side config
│       │   └── custom             # Custom training scripts and utilities for Swarm
│       │       ├── mlflow_receiver.py       # MLFlow integration for Swarm training
│       │       ├── model_persistor.py       # Model checkpoint handling
│       │       ├── split_data.py            # Data partitioning script (Swarm version)
│       │       └── train_runfile_nvflare.py # Main training script used in Swarm job
│       └── meta.conf             # Metadata for running the Swarm job in NVFLARE
├── README.md                     # Main project documentation and usage guide
└── requirements_nvflare.txt     # Required Python packages for running NVFLARE training jobs
     
```
##  How to Use

### 1. Clone the Repository and Install Dependencies

```bash
git clone -b tf2.19_nvflare https://github.com/ai4os-hub/phyto-plankton-classification.git
cd phyto-plankton-classification/nvflare
pip install -r requirements_nvflare  
cd ..
pip install --ignore-installed blinker -e .

```

## ⚠️ Virtual Environment Recommendation

It is **strongly recommended** to use a virtual environment (e.g., `venv`) when setting up and running this project to avoid dependency conflicts.

---

## 2. MLFlow Configuration

This setup uses an **MLFlow instance** (from ai4eosc) to track experiments. To avoid errors and ensure compatibility, update the MLFlow tracking URI in the following file:

```bash
jobs/decorator/app/config/config_fed_server.conf
```

Specifically, modify the component with the ID:

```bash
mlflow_receiver_with_tracking_uri
```



Replace the URI with the address of your **local MLFlow instance** or a **private MLFlow server**.

If your MLFlow server requires authentication, export the following environment variables in your terminal **before running the training**:
```bash
export MLFLOW_TRACKING_USERNAME='your_username'
export MLFLOW_TRACKING_PASSWORD='your_password'
```


---

## 3. Running the Federated Training

###  Step 1: Activate the Virtual Environment
```bash
source /path/to/venv/bin/activate
```

###  Step 2: Run the NVFLARE Simulator

**For FedAvg:** Execte the following commnad in the terminal:
```bash
nvflare simulator -n 2 -t 2 ./jobs/FedAvg -w FedAvg_workspace

```


**For Swarm Learning:**

```bash
nvflare simulator -n 2 -t 2 ./jobs/Swarm -w swarm_workspace
 

```
 



---

##  More Information

For a complete guide on training the phyto-plankton classification model and additional project details, please refer to the main `README.md` of the repository.

---

 

 
