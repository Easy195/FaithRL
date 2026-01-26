# PPO Training with PRM Integration (Verl Framework)

This repository contains the implementation for a PPO training pipeline using the **Verl** framework, integrated with a **Process Reward Model (PRM)**.

---

## 📂 Project Structure

| File/Directory | Description |
| :--- | :--- |
| `run.bash` | **Main Execution Entry**: The final script to trigger the training process. |
| `examples/run.sh` | **Example Script**: Shell script providing training configurations. |
| `pipeline.py` | **Deployment Script**: Deploys the PRM model (must run before training). |
| `verl/trainer/ppo/my_reward.py` | **Reward Logic**: Custom PRM reward function implementation. |
| `verl/trainer/ppo/ray_trainer.py` | **Core Trainer**: Ray-based PPO trainer implementation. |
| `verl/trainer/main_ppo.py` | **Main Entry**: Orchestration logic for the PPO training loop. |
| `mydata/train.parquet` | **Dataset**: Default training data in Parquet format. |

---

## 🚀 Workflow & Usage

[Image of PPO Reinforcement Learning with Process Reward Model architecture]

Follow these steps in order to start the training:

### 1. Data Preparation
Ensure your training data is located at the following path:
* **Path**: `mydata/train.parquet`

### 2. PRM Model Deployment
Before starting the training, you must deploy the PRM service to handle reward requests.
python3 pipeline.py

### 3. Start Training
Once the PRM service is live, execute the main bash script to initiate the PPO training process:
bash run.bash
```
