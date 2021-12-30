#!/bin/bash


# template:
# ~.../bin/python main.py (base arguments, e.g. --env_name) (il/offline algorithm, e.g. 'ValueDICE',) (specific arguments used in the chosen algorithm)

#~/anaconda3/envs/il/bin/python main.py --env_name 'Hopper-v2' --seed 0 --expert_algo 'valuedice' --total_expert_trajs 40 \
# --cuda --absorbing --norm_obs --deterministic_eval 'ValueDICE'
~/anaconda3/envs/il/bin/python main.py --env_name 'HalfCheetah-v2' --seed 0 --expert_algo 'valuedice' --total_expert_trajs 40 \
 --cuda --absorbing --norm_obs --deterministic_eval 'ValueDICE'
~/anaconda3/envs/il/bin/python main.py --env_name 'Walker2d-v2' --seed 0 --expert_algo 'valuedice' --total_expert_trajs 40 \
 --cuda --absorbing --norm_obs --deterministic_eval 'ValueDICE'
~/anaconda3/envs/il/bin/python main.py --env_name 'Ant-v2' --seed 0 --expert_algo 'valuedice' --total_expert_trajs 40 \
 --cuda --absorbing --norm_obs --deterministic_eval 'ValueDICE'

~/anaconda3/envs/il/bin/python main.py --env_name 'Hopper-v2' --seed 0 --expert_algo 'iq_learn' --total_expert_trajs 25 \
 --cuda --absorbing --norm_obs --deterministic_eval 'ValueDICE'
 ~/anaconda3/envs/il/bin/python main.py --env_name 'HalfCheetah-v2' --seed 0 --expert_algo 'iq_learn' --total_expert_trajs 25 \
 --cuda --absorbing --norm_obs --deterministic_eval 'ValueDICE'
~/anaconda3/envs/il/bin/python main.py --env_name 'Walker2d-v2' --seed 0 --expert_algo 'iq_learn' --total_expert_trajs 25 \
 --cuda --absorbing --norm_obs --deterministic_eval 'ValueDICE'
~/anaconda3/envs/il/bin/python main.py --env_name 'Ant-v2' --seed 0 --expert_algo 'iq_learn' --total_expert_trajs 25 \
 --cuda --absorbing --norm_obs --deterministic_eval 'ValueDICE'
