#!/bin/bash


~/anaconda3/envs/il/bin/python main.py env=Hopper-v2 method=ValueDICE seed=0 expert.expert_algo='valuedice' \
 expert.total_expert_trajs=40 expert.num_trajs=10
#~/anaconda3/envs/il/bin/python main.py env=HalfCheetah-v2 method=ValueDICE seed=0 expert.expert_algo='valuedice' \
# expert.total_expert_trajs=40 expert.num_trajs=10
#~/anaconda3/envs/il/bin/python main.py env=Ant-v2 method=ValueDICE seed=0 expert.expert_algo='valuedice' \
# expert.total_expert_trajs=40 expert.num_trajs=10
#~/anaconda3/envs/il/bin/python main.py env=Walker2d-v2 method=ValueDICE seed=0 expert.expert_algo='valuedice' \
# expert.total_expert_trajs=40 expert.num_trajs=10


