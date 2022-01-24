from il_offline_rl.algo.base_algo import BaseAlgo
from il_offline_rl.model import AC_agent, Q_agent, SAC_agent
from torch import nn
import torch
import numpy as np



class IQ_Learn(BaseAlgo):
    def __init__(
            self, args, log_dir, eval_env, expert_dataset, device, file_text='log.txt',
            env=None, online_replaybuffer=None):

        super(IQ_Learn, self).__init__(
            log_dir=log_dir, eval_env=eval_env, eval_log_interval=args.eval_log_interval,
            deterministic_eval=args.deterministic_eval, file_txt=file_text, eval_num_trajs=args.eval_num_trajs,
            env=env, online_replaybuffer=online_replaybuffer)

        agent_kwargs = {'hidden_size': args.agent.hidden_size,
                        'policy_dist': args.agent.policy_dist,
                        'acti_fn': eval(args.agent.acti_fn),
                        'is_mlp_base': args.agent.is_mlp_base,
                        'is_V_critic': args.agent.is_V_critic,
                        'is_double_Q': args.agent.is_double_Q}

        self.absorbing = args.method.absorbing
        input_dim = self.obs_shape[0]
        if self.absorbing: # have asserted that not (is_atari and absorbing)
            self.absorbing_per_episode = args.method.absorbing_per_episode
            input_dim += 1
        acs_dim =  self.action_space.n if self.is_discrete else self.action_space.shape[0]

        if self.is_discrete:
            # self.agent = Q_agent()
            pass
        else:
            self.agent = SAC_agent(input_dim, acs_dim, **agent_kwargs)
            self.agent.to(device)

        # init alpha temperature
        self.log_alpha = torch.tensor(np.log(args.agent.init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -acs_dim

        # init optimizer
        self.critic_optimizer = torch.optim.Adam(params=self.agent.get_critic_params(), lr=args.agent.critic_lr)
        self.actor_optimizer = torch.optim.Adam(params=self.agent.get_actor_params(), lr=args.agent.actor_lr)
        self.log_alpha_optimizer = torch.optim.Adam(params=[self.log_alpha], lr=args.agent.alpha_lr)

        # expert dataset handler
        self.expert_dataset = expert_dataset
        self.iter_expert_dataset = iter(self.expert_dataset)

        # other hyperparameters
        self.discount = args.discount
        self.batch_size = args.method.batch_size
        self.critic_tau = args.agent.critic_tau
        self.learnable_temperature = args.agent.learnable_temperature

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update(self):

        # sample expert batch
        try:
            expert_obs, expert_act, expert_next_obs = self.iter_expert_dataset.__next__()
        except:
            self.iter_expert_dataset = iter(self.expert_dataset)
            expert_obs, expert_act, expert_next_obs = self.iter_expert_dataset.__next__()

        # sample replay buffer batch, original size: (bs, dim)
        rb_obs, rb_act, rb_next_obs, _ = self.online_rb.sample(self.batch_size)


        _, expert_pi_act, expert_pi_act_log_probs = self.agent.get_action(expert_obs)
        expert_obs_pi_act = torch.cat((expert_obs, expert_pi_act), dim=1)
        expert_Q_obs_pi_act = self.agent.get_value(expert_obs_pi_act)
        expert_V_obs_pi_act = expert_Q_obs_pi_act - self.alpha.detach() * expert_pi_act_log_probs
        v0_loss = expert_V_obs_pi_act.mean()

        expert_obs_act = torch.cat((expert_obs, expert_act), dim=1)
        expert_Q_obs_act = self.agent.get_value(expert_obs_act)

        with torch.no_grad():
            _, expert_pi_next_act, expert_pi_next_act_log_probs = self.agent.get_action(expert_next_obs)
            expert_next_obs_pi_act = torch.cat((expert_next_obs, expert_pi_next_act), dim=1)
            expert_Q_next_obs_pi_act = self.agent.get_target_value(expert_next_obs_pi_act)
            evpert_V_next_obs_pi_act = expert_Q_next_obs_pi_act = self.alpha.detach() * expert_pi_next_act_log_probs
            y = done # todo: storage modify expert_done









        # v0_loss = self.agent.get_V(expert_obs, self.alpha)









