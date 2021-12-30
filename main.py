import os
import argparse
import torch
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.utils import make_dir
from a2c_ppo_acktr.storage import ExpertStorage, ReplayBuffer
from stable_baselines3.common.utils import set_random_seed

from a2c_ppo_acktr.algo import ValueDICE, specific_algo_params


def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    torch.set_num_threads(2)

    parser = argparse.ArgumentParser(description='key arguments for various imitation learning algorithms',
                                     add_help=False)
    # common arguments
    parser.add_argument('--env_name', type=str, default='Hopper-v2')
    parser.add_argument('--discount', type=float, default=0.99, help='discount factor used in RL')
    parser.add_argument('--gpu_idx', type=int, default=0, help='gpu index')
    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use gpu')
    parser.add_argument('--seed', type=int, default=0, help='seed for env and choice of trajectories')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of training samples')

    parser.add_argument('--eval_num_processes', type=int, default=5, help='num of parallel evaluation envs')
    parser.add_argument('--num_processes', type=int, default=1, help='num of parallel online interactive envs')

    parser.add_argument('--expert_algo', type=str, default='valuedice',
                        help='algorithm or projects used for sampling expert data (e.g. ppo, valuedice, iq_learn)')

    parser.add_argument('--total_expert_trajs', type=int, default=40, help='total number of sampled expert trajs')
    parser.add_argument('--eval_num_trajs', type=int, default=10, help='number of trajectories when evaluating')
    parser.add_argument('--num_trajs', type=int, default=10, help='number of expert trajectories used for imitation learning')

    parser.add_argument('--eval_log_interval', type=int, default=2000, help='interval between two loggings of evaluation')
    parser.add_argument('--update_log_interval', type=int, default=5000, help='interval between two parameter updates')

    parser.add_argument('--expert_file_path', type=str, default='/home/zy/zz/all_logs/rl-w-data/expert_trajs',
                        help='path to storage all the expert trajectories')
    parser.add_argument('--log_dir', type=str, default='/home/zy/zz/all_logs/rl-w-data/results')

    parser.add_argument('--max_episode_steps', type=int, default=1e3,
                        help='max steps of each episode, default: 1000 for mujoco')
    parser.add_argument('--max_timesteps', type=int, default=5e5, help='max training timesteps')
    parser.add_argument('--start_training_steps', type=int, default=1e3,
                        help='number of samples before training agent(for online IL)')

    parser.add_argument('--absorbing', action='store_true', default=False, help='whether using absorbing state')
    parser.add_argument('--absorbing_per_episode', type=int, default=10,
                        help='if absorbing, max number of absorbing states which can be added in one episode')

    parser.add_argument('--norm_obs', action='store_true', default=False, help='whether normalizing obs')
    parser.add_argument('--deterministic_eval', action='store_true', default=False,
                        help='whether deteministic or stochastic policy when implementing evaluation')


    # specific algo arguments augment
    args = specific_algo_params(parser)

    # device
    if args.cuda and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_idx)
        device = torch.device('cuda:{}'.format(args.gpu_idx))
    else:
        device = torch.device('cpu')

    # set seed
    set_random_seed(args.seed, using_cuda=True)

    # log dir of training result (.../rl-w-data/results/ValueDICE/Hopper-v3/xxx_xxx_seed=x)
    log_dir = os.path.join(args.log_dir, args.il_algo, args.env_name,
                           f'num_trajs={args.num_trajs}_absorbing={int(args.absorbing)}_'
                           f'norm_obs={int(args.norm_obs)}_expert_algo={args.expert_algo}_seed={args.seed}')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    else:
        print('##### this configure has done #####')
        return 0

    # load expert dataset
    expert_file_path = os.path.join(args.expert_file_path, args.expert_algo, args.env_name)
    print('######### load expert dataset... ##########')
    expert_storage = ExpertStorage(expert_file_path, args.env_name, args.seed, args.total_expert_trajs, args.num_trajs,
                                   args.batch_size, device, obs_norm=args.norm_obs, absorbing=args.absorbing,
                                   max_episode_length=args.max_episode_steps)

    expert_dataset = expert_storage.load_trasition_dataset()
    obs_mean_var = expert_storage.get_mean_var() if args.norm_obs else None

    # make online env(single env for now) and eval_env(can multi envs parallel)
    is_atari = True if 'NoFrameskip' in args.env_name else False
    assert not (is_atari and args.absorbing), 'Atari envs don\'t support absorbing state wrapper.'

    env = make_vec_envs(args.env_name, args.seed, 1, 0.99, device, True, absorbing_state=args.absorbing,
                        normalize_obs=args.norm_obs, fixed_obs_rms=obs_mean_var) # mujoco default 1e3 max_episode_steps
    eval_env = make_vec_envs(args.env_name, args.seed+10, args.eval_num_processes, 0.99, device, True,
                             absorbing_state=args.absorbing, normalize_obs=args.norm_obs, fixed_obs_rms=obs_mean_var)

    print('######### initial replaybuffer for online setting ########')
    # True obs space space (modify obs space shape if necessary)
    real_obs_shape = env.observation_space.shape
    if args.absorbing:
        real_obs_shape = (real_obs_shape[0]+1,)
    action_space = env.action_space

    online_replaybuffer = ReplayBuffer(int(args.max_timesteps*2), real_obs_shape, action_space, device)

    print('######### begin training phase ########')
    if args.il_algo == 'ValueDICE':
        agent = ValueDICE(args, log_dir, eval_env, args.hidden_size, expert_dataset, device,
                          env=env, online_replaybuffer=online_replaybuffer)
        agent.train()
    else:
        raise ValueError('Don\'t support {} algorithm for now'.format(args.il_algo))


    env.close()
    eval_env.close()



if __name__ == "__main__":
    main()