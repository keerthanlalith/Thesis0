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
[OpenAI Baselines](https://github.com/openai/baselines) is used to obtain these trajectories. 
Collecting the data consists of two steps: 
1) training the expert and 
2) running the learned policy and saving the observed state trajectories to disk.

```Shell
# 1) Train expert
python train_expert/train_cartpole.py

# 2) Collect state trajectories 
python enjoy_cartpole_custom.py
# 50000 steps are collected for training data
# 1000 steps are collected for test data
```

Once done running, the expert policy from step 1 is written to [Data/cartpole.pkl](https://github.com/keerthanlalith/Thesis0/tree/main/Data/). Then, step 2 loads and runs the policy and saves the observed states, actions, next states  and difference(between the current state and the nextstate) to [Data](https://github.com/keerthanlalith/Thesis0/tree/main/Data). 


## Training Next state predictor/ autoencoder

Given the current state, the 2 layer NN is used to predict the next state
The NN is tained in a spervised learning manner. The current state and Next state for training taken from State.npy, NState.npy [Data](https://github.com/keerthanlalith/Thesis0/tree/main/Data). The autoendoer is tested on test data taken from TState.npy, TNState.npy
The NN stucture is defined in the AE.py, with leakyRELU activation fucntion

Each NN is trained for 50000 iterations, with batch size of 64, learning rate = 0.00005

to train the NN, run

```Shell
python ae1.py
#utilises the autoencoder1 stucture defiend in AE.py
# similary
python ae2.py
#utilises the autoencoder2 stucture defiend in AE.py
```

The NN prints predicted diff, true Test diff,  Difference(abs difference between predicted and true)
```Shell
AE diff output, Normalised Test diff,  Difference
[-0.003946  0.270008 -0.004468 -0.343998] [-4.43000e-04  1.94754e-01  9.00000e-05 -2.84683e-01] [0.003503 0.075253 0.004557 0.059315]
[ 0.005893  0.201137 -0.009503 -0.255153] [ 0.003452  0.194755 -0.005604 -0.284657] [0.002441 0.006382 0.003899 0.029504]
[ 0.015417  0.13489  -0.01275  -0.179717] [ 0.007347  0.194843 -0.011297 -0.286473] [0.00807  0.059953 0.001453 0.106755]
[ 0.012683  0.074842 -0.014628 -0.086911] [ 0.011244  0.195009 -0.017027 -0.290089] [0.001439 0.120167 0.002398 0.203178]
[ 0.007159 -0.128288 -0.009663  0.221772] [ 0.015144 -0.195006 -0.022828  0.289906] [0.007986 0.066718 0.013166 0.068134]
[ 0.006812 -0.162887 -0.005695  0.248859] [ 0.011244 -0.194677 -0.01703   0.282568] [0.004432 0.031789 0.011335 0.033709]
[ 0.005747 -0.183409 -0.004876  0.26732 ] [ 0.007351 -0.194407 -0.011379  0.276961] [0.001604 0.010997 0.006503 0.009641]
[ 0.00427  -0.193931 -0.003931  0.277663] [ 0.003462 -0.194217 -0.00584   0.273156] [0.000807 0.000286 0.001909 0.004508]
[ 0.002254 -0.193745 -0.002791  0.27936 ] [-0.000422 -0.194118 -0.000377  0.271188] [0.002676 0.000373 0.002414 0.008172]
[-0.005473 -0.10213  -0.00061   0.172012] [-0.004304  0.196003  0.005047 -0.312827] [0.001169 0.298133 0.005657 0.484839]
------
[-0.003946  0.270008 -0.004468 -0.343998] [-4.43000e-04  1.94754e-01  9.00000e-05 -2.84683e-01] [0.003503 0.075253 0.004557 0.059315]
[ 0.005893  0.201137 -0.009503 -0.255153] [ 0.003452  0.194755 -0.005604 -0.284657] [0.002441 0.006382 0.003899 0.029504]
[ 0.015417  0.13489  -0.01275  -0.179717] [ 0.007347  0.194843 -0.011297 -0.286473] [0.00807  0.059953 0.001453 0.106755]
[ 0.012683  0.074842 -0.014628 -0.086911] [ 0.011244  0.195009 -0.017027 -0.290089] [0.001439 0.120167 0.002398 0.203178]
[ 0.007159 -0.128288 -0.009663  0.221772] [ 0.015144 -0.195006 -0.022828  0.289906] [0.007986 0.066718 0.013166 0.068134]
[ 0.006812 -0.162887 -0.005695  0.248859] [ 0.011244 -0.194677 -0.01703   0.282568] [0.004432 0.031789 0.011335 0.033709]
[ 0.005747 -0.183409 -0.004876  0.26732 ] [ 0.007351 -0.194407 -0.011379  0.276961] [0.001604 0.010997 0.006503 0.009641]
[ 0.00427  -0.193931 -0.003931  0.277663] [ 0.003462 -0.194217 -0.00584   0.273156] [0.000807 0.000286 0.001909 0.004508]
[ 0.002254 -0.193745 -0.002791  0.27936 ] [-0.000422 -0.194118 -0.000377  0.271188] [0.002676 0.000373 0.002414 0.008172]
[-0.005473 -0.10213  -0.00061   0.172012] [-0.004304  0.196003  0.005047 -0.312827] [0.001169 0.298133 0.005657 0.484839]
------
[2.08278047e-06 8.63954016e-03 2.84632155e-06 1.90368683e-02] [2.08278047e-06 8.63954016e-03 2.84632155e-06 1.90368683e-02]
```
lastly it prints the mean square error state vise on the test data

to train all NN structures, run

```Shell
./testall.sh
```

The resluts of the script are in [Results](https://github.com/keerthanlalith/Thesis0/tree/main/Results), data1.txt,data2.txt ...respectively
the mean square error state vise for all auto encoder stuctures are on present in [Dataall.txt](https://github.com/keerthanlalith/Thesis0/blob/main/Dataall.txt)
