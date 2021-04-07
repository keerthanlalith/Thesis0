# Thesis0

## Requirements 
This implementation has been tested with Python 3.5 on Ubuntu 18.04. 

```Shell
# 1) Clone repository 
git clone https://https://github.com/keerthanlalith/Thesis0
git clone https://github.com/openai/baselines.git
cd Thesis0

# 2) Install requirements
pip install -r requirements.txt
cd baselines
pip install -e .
``` 

## Collecting expert data 
[OpenAI Baselines](https://github.com/openai/baselines) to obtain these trajectories. 
Collecting the data consists of two steps: 
1) training the expert and 
2) running the learned policy and saving the observed state trajectories to disk.

```Shell
# 1) Train expert
python train_expert/train_cartpole.py

# 2) Collect state trajectories 
python enjoy_cartpole_custom.py
```

Once done running, the expert policy from step 1 is written to [Data/cartpole.pkl](https://github.com/keerthanlalith/Thesis0/tree/main/Data/). Then, step 2 loads and runs the policy and saves the observed states, actions, next states  and difference(between the current state and the nextstate) to [Data](https://github.com/keerthanlalith/Thesis0/tree/main/Data). 

