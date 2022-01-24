# rl-w-data

This project aims to reproduce some currently popular **imitation learning** & **offline reinforcement** **learning** algorithms, and compare them within the same framework.



### done & planning

- [x] ValueDICE 
- [ ] iq-learn 
- [ ] sqil 



### Implementation
This project works with Python 3.7, and before starting the implementation, please check and install the necessary packages listed in  `./requirements.txt`:

```bash
pip install -r requirements.txt
```

Quick start (ensure that `./run_il_script.sh` is executable, and within it, you can choose the algorithms that you want to run):

```bash
./run_il_script.sh
```




### Reference

- [Imitation Learning via Off-policy Distribution Matching](https://arxiv.org/abs/1912.05032v1) [[code](https://github.com/google-research/google-research/tree/master/value_dice)]
- [IQ-Learn: Inverse soft-Q learning fro Imitation](https://arxiv.org/abs/2106.12142v2) [[code](https://github.com/Div99/IQ-Learn)]
- [SQIL: Imitation Learning via Reinforcement Learning with Sparse Rewards](https://arxiv.org/abs/1905.11108v3) 

- [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)

- [stable-baseline3](https://github.com/DLR-RM/stable-baselines3)

