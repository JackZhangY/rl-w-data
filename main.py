import os
import argparse
import torch
from il_offline_rl.envs import make_vec_envs
from il_offline_rl.utils import make_dir
from il_offline_rl.storage import ExpertStorage, ReplayBuffer
from stable_baselines3.common.utils import set_random_seed
import hydra
from omegaconf import DictConfig, OmegaConf

from il_offline_rl.algo import ValueDICE


@hydra.main(config_path='il_offline_rl/config', config_name='base.yaml')
def main(args):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    torch.set_num_threads(2)


    config_info = OmegaConf.to_yaml(args)

    # device
    if args.cuda and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_idx)
        device = torch.device('cuda:{}'.format(args.gpu_idx))
    else:
        device = torch.device('cpu')

    # set seed
    set_random_seed(args.seed, using_cuda=True)

    # log dir of training result (.../rl-w-data/results/Hopper-v3/ValueDICE/xxx_xxx_seed=x)
    log_dir = os.path.join(args.log_dir, args.env.env_name, args.method.method_name,
                           f'num_trajs={args.expert.num_trajs}_absorbing={int(args.method.absorbing)}_'
                           f'norm_obs={int(args.method.norm_obs)}_expert_algo={args.expert.expert_algo}_seed={args.seed}')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    else:
        print('##### this configure has done #####')
        return 0

    # load expert dataset
    expert_file_path = os.path.join(args.expert.expert_file_path, args.expert.expert_algo, args.env.env_name)
    print('######### load expert dataset... ##########')
    expert_storage = ExpertStorage(
        expert_file_path, args.env.env_name, args.seed, args.expert.total_expert_trajs, args.expert.num_trajs,
        args.method.batch_size, device, obs_norm=args.method.norm_obs, absorbing=args.method.absorbing,
        max_episode_length=args.env.max_episode_steps)

    expert_dataset = expert_storage.load_trasition_dataset()
    obs_mean_var = expert_storage.get_mean_var() if args.method.norm_obs else None

    # make online env(single env for now) and eval_env(can multi envs parallel)
    is_atari = True if 'NoFrameskip' in args.env.env_name else False
    assert not (is_atari and args.method.absorbing), 'Atari envs don\'t support absorbing state wrapper.'

    env = make_vec_envs(
        args.env.env_name, args.seed, 1, 0.99, device, True, absorbing_state=args.method.absorbing,
        normalize_obs=args.method.norm_obs, fixed_obs_rms=obs_mean_var) # mujoco default 1e3 max_episode_steps
    eval_env = make_vec_envs(
        args.env.env_name, args.seed+10, args.eval.eval_num_processes, 0.99, device, True,
        absorbing_state=args.method.absorbing, normalize_obs=args.method.norm_obs, fixed_obs_rms=obs_mean_var)

    print('######### initial replaybuffer for online setting ########')
    # True obs space space (modify obs space shape if necessary)
    real_obs_shape = env.observation_space.shape
    if args.method.absorbing:
        real_obs_shape = (real_obs_shape[0]+1,)
    action_space = env.action_space

    online_replaybuffer = ReplayBuffer(int(args.method.max_timesteps*2), real_obs_shape, action_space, device)

    print('######### begin training phase ########')
    if args.method.method_name == 'ValueDICE':
        agent = ValueDICE(args, log_dir, eval_env,  expert_dataset, device,
                          env=env, online_replaybuffer=online_replaybuffer)
        agent.save_config(config_info)
        agent.train()
    else:
        raise ValueError('Don\'t support {} algorithm for now'.format(args.method.method_name))


    env.close()
    eval_env.close()



if __name__ == "__main__":
    main()