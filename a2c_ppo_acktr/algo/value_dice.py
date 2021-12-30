from a2c_ppo_acktr.model import Policy, Separate_AC_Net
from torch import nn
import torch
from torch.autograd import Variable, grad
import numpy as np
from .base_algo import Logger, BaseAlgo
from collections import deque

EPS = np.finfo(np.float32).eps

class ValueDICE(BaseAlgo):
    def __init__(self, args, log_dir, eval_env, hidden_size, expert_dataset, device, file_text='log.txt',
                 env=None, online_replaybuffer=None):


        super(ValueDICE, self).__init__(log_dir=log_dir, eval_env=eval_env, eval_log_interval=args.eval_log_interval,
                                  deterministic_eval=args.deterministic_eval, file_txt=file_text,
                                  eval_num_trajs=args.eval_num_trajs, env=env, online_replaybuffer=online_replaybuffer)

        # init nu net and pi net, only for mujoco, todo: for atari envs

        policy_kwargs = {'hidden_size' : hidden_size, 'acti_fn' : nn.ReLU()}
        if not self.is_atari:
            policy_kwargs['action_dim'] = self.action_space.shape[0]

        self.absorbing = args.absorbing
        if self.absorbing: # have asserted that not (is_atari and absorbing)
            self.absorbing_per_episode = args.absorbing_per_episode
            #  VecAbsorbingState don't change the original obs sapce shape, so should modify it here
            self.obs_shape = (self.obs_shape[0]+1,)

        self.agent = Separate_AC_Net(self.obs_shape, self.action_space, base_kwargs=policy_kwargs)
        self.agent.to_device(device)

        # init optimizer
        self.critic_optimizer = torch.optim.Adam(params=self.agent.get_critic_params(), lr=args.nu_lr)
        self.actor_optimizer = torch.optim.Adam(params=self.agent.get_actor_params(), lr=args.actor_lr)

        # expert dataset handler
        self.expert_dataset = expert_dataset
        self.iter_expert_dataset = iter(self.expert_dataset)

        # other hyperparameter
        self.discount = args.discount
        self.batch_size = args.batch_size

        self.replay_reg = args.replay_regularization
        self.nu_reg = args.nu_regularization

        self.updates_per_step = args.updates_per_step
        self.update_log_interval = args.update_log_interval

        self.num_random_actions = args.num_random_actions
        self.start_training_steps = args.start_training_steps
        self.max_episode_steps = args.max_episode_steps
        self.max_timesteps = args.max_timesteps
        self.update_steps= 0

        self.device = device

        self.args_info(args)


    def train(self):
        ##### env has only one process #####
        obs = self.env.reset() # tensor: (1, dim)
        train_episode_window = deque(maxlen=10)
        total_train_episodes = 0
        next_obs = obs
        dones = [False]* self.env.num_envs
        truncated_dones = [True]* self.env.num_envs
        infos = [None]* self.env.num_envs
        episode_timesteps = np.zeros((self.env.num_envs,), dtype=np.float32)

        while self.total_timesteps < self.max_timesteps:
            if self.total_timesteps % self.eval_log_interval == 0:
                eval_result = self.evaluate()
                self.add_scalar('returns/eval', eval_result, self.total_timesteps)

            #### todo: start batch process if multiprocessing env
            if dones[0]:
                if 'episode' in infos[0].keys():
                    train_episode_window.append(infos[0]['episode']['r'])
                episode_timesteps[0] = 0
                total_train_episodes += 1
                if total_train_episodes % 5 == 0:
                    self.add_scalar('returns/train', np.mean(train_episode_window), self.total_timesteps)

            if self.total_timesteps < self.num_random_actions:
                act = self.env.action_space.sample() # np: (dim,)
                act = torch.tensor(np.expand_dims(act, axis=0)).to(self.device) # (1, dim)
            else:
                act = self.agent.act(obs, deterministic=True) # deterministic action (1, dim)
                act = torch.clamp(torch.normal(0, 0.1, size=act.size()).to(self.device) + act, -1., 1.).detach()

            next_obs, _, dones, infos = self.env.step(act)

            truncated_dones[0] = dones[0] and episode_timesteps[0] + 1 == self.max_episode_steps

            if dones[0] and not truncated_dones[0] and self.absorbing:
                # if done, next_obs will be the initial obs for the next episode
                absorbing_obs = self.env.get_absorbing_state()# tensor: (dim,)
                self.online_rb.add_batch(obs[0], act[0], absorbing_obs)

                for i in range(self.absorbing_per_episode):

                    if episode_timesteps[0]+i < self.max_episode_steps:
                        absorbing_obs = self.env.get_absorbing_state()
                        random_act = self.env.action_space.sample() # (dim,)
                        random_act = torch.tensor(random_act).to(self.device)
                        next_absorbing_obs = self.env.get_absorbing_state()

                        self.online_rb.add_batch(absorbing_obs, random_act, next_absorbing_obs)
            else:
                if truncated_dones[0]:
                    real_next_obs = infos[0]['terminal_observation']
                    real_next_obs = np.concatenate([real_next_obs, np.array([0.])], axis=0).astype(np.float32)
                    real_next_obs = torch.tensor(real_next_obs).to(self.device)
                    self.online_rb.add_batch(obs[0], act[0], real_next_obs)
                else:
                    self.online_rb.add_batch(obs[0], act[0], next_obs[0])

            ### end batch process if multiprocessing env

            episode_timesteps += 1 # np broadcast
            self.total_timesteps += 1 * self.env.num_envs

            obs = next_obs

            if self.total_timesteps >= self.start_training_steps:
                for _ in range(self.updates_per_step):
                    self.update()



    # todo: only for mujoco now, should further modify for Atari envs
    def update(self):

        # sample expert batch
        try:
            expert_obs, expert_act, expert_next_obs = self.iter_expert_dataset.__next__()
        except:
            self.iter_expert_dataset = iter(self.expert_dataset)
            expert_obs, expert_act, expert_next_obs = self.iter_expert_dataset.__next__()

        expert_initial_obs = expert_obs

        # sample replay buffer batch, original size: (bs, dim)
        rb_obs, rb_act, rb_next_obs, _ = self.online_rb.sample(self.batch_size)

        _, expert_next_act, _ = self.agent(expert_next_obs)
        _, rb_next_act, _ = self.agent(rb_next_obs)
        _, policy_initial_act, _ = self.agent(expert_initial_obs)

        # construct nu net input
        expert_init_obs_act = torch.cat((expert_initial_obs, policy_initial_act), dim=1)
        expert_next_obs_act = torch.cat((expert_next_obs, expert_next_act), dim=1)
        expert_obs_act = torch.cat((expert_obs, expert_act), dim=1)

        rb_next_obs_act = torch.cat((rb_next_obs, rb_next_act), dim=1)
        rb_obs_act = torch.cat((rb_obs, rb_act), dim=1)

        # input (obs, act) into nu net
        expert_init_nu = self.agent.get_value(expert_init_obs_act)
        expert_next_nu = self.agent.get_value(expert_next_obs_act)
        expert_nu = self.agent.get_value(expert_obs_act)

        rb_next_nu = self.agent.get_value(rb_next_obs_act)
        rb_nu = self.agent.get_value(rb_obs_act)

        expert_inv_Bellman = expert_nu - self.discount * expert_next_nu
        rb_inv_Bellman = rb_nu - self.discount * rb_next_nu

        # linear loss
        linear_loss_expert = torch.mean(expert_init_nu * (1 - self.discount))
        linear_loss_rb = torch.mean(rb_inv_Bellman)

        linear_loss = linear_loss_expert * (1 - self.replay_reg) + linear_loss_rb * self.replay_reg

        # nonlinear loss
        mix_inv_Bellman = torch.cat([expert_inv_Bellman, rb_inv_Bellman], dim=0)
        mix_weights = torch.cat(
            [torch.ones(expert_inv_Bellman.shape) * (1 - self.replay_reg),
             torch.ones(rb_inv_Bellman.shape) * (self.replay_reg)], dim=0).to(self.device)

        mix_weights /= torch.sum(mix_weights)

        with torch.no_grad():
            weighted_softmax_mix_weights = self.weighted_softmax(mix_inv_Bellman, mix_weights)

        non_linear_loss = torch.sum(weighted_softmax_mix_weights.detach() * mix_inv_Bellman)

        # total loss with gradient penalty and orthogonal regularization
        total_loss = non_linear_loss - linear_loss

        nu_grad_penalty = self.grad_penalty(expert_obs_act, rb_obs_act, expert_next_obs_act, rb_next_obs_act)
        orth_reg = self.orthogonal_regularization(self.agent.get_actor_params(True))

        nu_loss = total_loss + nu_grad_penalty * self.nu_reg
        pi_loss = -total_loss + orth_reg

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        nu_loss.backward(retain_graph=True, inputs=self.agent.get_critic_params())
        pi_loss.backward(inputs=self.agent.get_actor_params())

        self.critic_optimizer.step()
        self.actor_optimizer.step()

        # record the intermediate variable w.r.t update steps
        self.update_steps += 1

        if self.update_steps % self.update_log_interval == 0:
            # record nu(Q) value
            self.add_scalar('value_est/expert_nu', torch.mean(expert_nu), self.update_steps)
            self.add_scalar('value_est/rb_nu', torch.mean(rb_nu), self.update_steps)
            # record inverse Bellman operator
            self.add_scalar('inv_Bellman/expert_inv_Bellman', torch.mean(expert_inv_Bellman), self.update_steps)
            self.add_scalar('inv_Bellman/rb_inv_Bellman', torch.mean(rb_inv_Bellman), self.update_steps)
            # record loss
            self.add_scalar('loss/total_loss', total_loss.item(), self.update_steps)
            self.add_scalar('loss/grad_penalty', nu_grad_penalty.item(), self.update_steps)
            self.add_scalar('loss/orth_reg', orth_reg.item(), self.update_steps)


    def grad_penalty(self, expert_obs_act, rb_obs_act, expert_next_obs_act, rb_next_obs_act):
        alpha = torch.rand(expert_obs_act.size()[0], 1)
        alpha = alpha.expand_as(expert_obs_act).to(self.device)

        mix_obs_act = alpha * expert_obs_act + (1 - alpha) * rb_obs_act
        mix_next_obs_act = alpha * expert_next_obs_act + (1 - alpha) * rb_next_obs_act

        total_mix_obs_act = Variable(torch.cat([mix_obs_act, mix_next_obs_act], 0), requires_grad=True)
        total_q = self.agent.get_value(total_mix_obs_act)
        ones = torch.ones(total_q.size()).to(self.device)

        gradient = grad(
            outputs=total_q,
            inputs=total_mix_obs_act,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0] + EPS

        grad_penalty = (gradient.norm(2, dim=1) - 1).pow(2).mean()
        return grad_penalty


    def orthogonal_regularization(self, named_params_list, reg_coef=1e-4):
        reg = 0
        for name, param in named_params_list:
            # todo: only for the weights of linear layers (mujoco), not used for conv layers
            if 'weight' in name and param.requires_grad:
                prod = torch.matmul(param.T, param)
                reg += torch.sum(torch.square(prod * (1 - torch.eye(prod.shape[0])).to(self.device)))

        return reg * reg_coef

    def weighted_softmax(self, mix_inv_Bellman, mix_weights, dim=0):
        x = mix_inv_Bellman - torch.max(mix_inv_Bellman, dim=dim)[0]
        weighted_sm = mix_weights * torch.exp(x) / torch.sum(
            mix_weights * torch.exp(x), dim=dim, keepdim=True)
        return weighted_sm


























