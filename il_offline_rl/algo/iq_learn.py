from il_offline_rl.algo.base_algo import BaseAlgo
from il_offline_rl.model import AC_agent, Q_agent, SAC_agent
from torch import nn
from collections import deque
from torch.autograd import Variable, grad
import torch
import numpy as np


EPS = np.finfo(np.float32).eps

class IQ_Learn(BaseAlgo):
    def __init__(
            self, args, log_dir, eval_env, expert_dataset, device, file_text='log.txt',
            env=None, online_replaybuffer=None):

        super(IQ_Learn, self).__init__(
            log_dir=log_dir, eval_env=eval_env, eval_log_interval=args.eval.eval_log_interval,
            deterministic_eval=args.eval.deterministic_eval, file_txt=file_text, eval_num_trajs=args.eval.eval_num_trajs,
            env=env, online_replaybuffer=online_replaybuffer)

        agent_kwargs = {'hidden_size': args.agent.hidden_size,
                        'policy_dist': args.agent.policy_dist,
                        'acti_fn': eval(args.agent.acti_fn),
                        'is_mlp_base': args.agent.is_mlp_base,
                        'is_V_critic': args.agent.is_V_critic,
                        'is_double_Q': args.agent.is_double_Q}

        self.absorbing = args.method.absorbing # no absorbing for IQ-Learn

        input_dim = self.obs_shape[0]
        if self.absorbing: # have asserted that not (is_atari and absorbing)
            self.absorbing_per_episode = args.method.absorbing_per_episode
            input_dim += 1
        acs_dim =  self.action_space.n if self.is_discrete else self.action_space.shape[0]

        if self.is_discrete:
            # todo: for atari envs
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
        self.device = device
        self.update_steps = 0
        self.discount = args.discount

        self.max_episode_steps = args.env.max_episode_steps

        self.batch_size = args.method.batch_size
        self.max_timesteps = args.method.max_timesteps
        self.num_random_steps = args.method.num_random_steps
        self.start_training_steps = args.method.start_training_steps
        self.use_target = args.method.use_target
        self.loss_type = args.method.loss_type
        self.grad_pen = args.method.grad_pen # list,
        self.regularize = args.method.regularize # list,
        self.action_gap_reg = args.method.action_gap_reg # list
        self.update_log_interval = args.method.update_log_interval

        self.critic_tau = args.agent.critic_tau
        self.updates_per_step = args.agent.updates_per_step
        self.target_critic_update_freq = args.agent.target_critic_update_freq
        self.learnable_temperature = args.agent.learnable_temperature

    @property
    def alpha(self):
        return self.log_alpha.exp()

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
                if total_train_episodes % 10 == 0:
                    self.add_scalar('returns/train', np.mean(train_episode_window), self.total_timesteps)

            if self.total_timesteps < self.num_random_steps:
                act = self.env.action_space.sample() # np: (dim,)
                act = torch.tensor(np.expand_dims(act, axis=0)).to(self.device) # (1, dim)
            else:
                act = self.agent.act(obs, deterministic=False) # (1, dim)
                act = torch.clamp(act, -1., 1.).detach()
                # act = torch.clamp(torch.normal(0, 0.1, size=act.size()).to(self.device) + act, -1., 1.).detach()

            next_obs, _, dones, infos = self.env.step(act)

            # add the online samples
            truncated_dones[0] = dones[0] and episode_timesteps[0] + 1 == self.max_episode_steps
            assert infos[0].get('TimeLimit.truncated', False) == truncated_dones[0], 'wrong truncated trajectory'

            real_next_obs = next_obs[0]
            if dones[0]:
                real_next_obs = torch.tensor(infos[0]['terminal_observation']).float().to(self.device)
            self.online_rb.add_batch(obs[0], act[0], real_next_obs, dones[0], truncated_dones[0])

            ### end batch process if multiprocessing env
            episode_timesteps += 1 # np broadcast
            self.total_timesteps += 1 * self.env.num_envs

            obs = next_obs

            if self.total_timesteps >= self.start_training_steps:
                for _ in range(self.updates_per_step):
                    self.update()


    def update(self):
        # sample expert batch
        try:
            # (bs, dim/1)
            expert_obs, expert_acs, expert_next_obs, expert_dones = self.iter_expert_dataset.__next__()
        except:
            self.iter_expert_dataset = iter(self.expert_dataset)
            expert_obs, expert_acs, expert_next_obs, expert_dones = self.iter_expert_dataset.__next__()

        # sample replay buffer batch, original size: (bs, dim) or rb_dones (bs, )
        rb_obs, rb_acs, rb_next_obs, rb_dones = self.online_rb.sample(self.batch_size)

        # concatenate expert and rb data
        both_obs = torch.cat((expert_obs, rb_obs), dim=0)
        both_acs = torch.cat((expert_acs, rb_acs), dim=0)
        both_next_obs = torch.cat((expert_next_obs, rb_next_obs), dim=0)
        both_dones = torch.cat((expert_dones, rb_dones.unsqueeze(1)), dim=0)

        #####################  UPDATE Critic #####################

        # 1st term of loss: -E_(d_expert)[Q(s,a)-gamma V(s')]
        both_obs_acs = torch.cat((both_obs, both_acs), dim=1)
        both_current_Q = self.agent.get_value(both_obs_acs)[0]

        if self.use_target:
            with torch.no_grad():
                both_next_v = self.agent.get_target_V(both_next_obs, alpha=self.alpha)
                both_Y = (1 - both_dones) * self.discount * both_next_v
        else:
            both_next_v = self.agent.get_V(both_next_obs, alpha=self.alpha)
            both_Y = (1 - both_dones) * self.discount * both_next_v

        both_r = (both_current_Q - both_Y)

        # additional Adv function
        if self.action_gap_reg[0]:
            # note: have target V, whether target Q?
            with torch.no_grad():
                both_target_v = self.agent.get_target_V(both_obs, alpha=self.alpha)
            scaling_A = -self.action_gap_reg[1]*(both_current_Q - both_target_v)
            both_r += scaling_A
        ####### end addition of expert scaling advantage function ######

        expert_r_loss = -torch.mean(both_r[:self.batch_size])
        total_loss = expert_r_loss

        # 2nd term of loss:
        if self.loss_type == 'v0':
            both_v = self.agent.get_V(both_obs, alpha=self.alpha)
            v0 = torch.mean(both_v[:self.batch_size])
            # v0 = self.agent.get_V(expert_obs, alpha=self.alpha).mean()
            rb_r_loss = (1 - self.discount) * v0

            if self.action_gap_reg[0]:
                rb_r_loss += torch.mean(scaling_A[self.batch_size:]) # still use non-expert data

            total_loss  += rb_r_loss
        elif self.loss_type == 'value':
            both_v = self.agent.get_V(both_obs, alpha=self.alpha)
            rb_r_loss = (both_v - both_Y).mean()
            # rb_r_loss = (self.agent.get_V(both_obs, alpha=self.alpha) - both_y).mean()

            if self.action_gap_reg[0]:
                rb_r_loss += torch.mean(scaling_A[self.batch_size:]) # still use non-expert data

            total_loss += rb_r_loss
        else:
            raise ValueError('no this loss type')


        #  gradient penalty
        if self.grad_pen[0]:
            expert_obs_acs, rb_obs_acs = torch.split(both_obs_acs, [self.batch_size, self.batch_size], dim=0)
            gp_loss = self.grad_penalty(expert_obs_acs, rb_obs_acs, self.grad_pen[1])
            total_loss += gp_loss

        # regularization for the learned reward of expert and policy (only expert data described in the original paper)
        if self.regularize[0]:
            # \Chi^2 divergence regularization
            chi2_loss = 1/(4 * self.regularize[1]) * (both_r**2).mean()
            total_loss += chi2_loss

        # update critic
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        self.critic_optimizer.step()

        #####################  UPDATE Actor #####################
        # actor loss
        _, both_actions, both_logprobs = self.agent.get_action(both_obs)
        both_obs_pi = torch.cat((both_obs, both_actions), dim=1)
        both_Q = self.agent.get_value(both_obs_pi)[0]
        actor_loss = (self.alpha.detach() * both_logprobs - both_Q).mean()

        # update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #####################  UPDATE Entropy Temperature #####################
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-both_logprobs - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.update_steps += 1
        #####################  UPDATE Target Critic #########################
        if self.update_steps % self.target_critic_update_freq == 0:
            self.agent.soft_update(self.critic_tau)

        #####################  Variable Logging #####################

        if self.update_steps % self.update_log_interval == 0:
            # record loss
            # self.add_scalar('loss/expert_r_loss', expert_r_loss.item(), self.update_steps)
            self.add_scalar('loss/total_loss', total_loss.item(), self.update_steps)
            self.add_scalar('loss/rb_r_loss', rb_r_loss.item(), self.update_steps)
            if self.grad_pen[0]:
                self.add_scalar('loss/grad_penalty', gp_loss.item(), self.update_steps)
            if self.regularize[0]:
                self.add_scalar('loss/chi2_loss', chi2_loss.item(), self.update_steps)
            # record reward (inverse bellman)
            self.add_scalar('inv_Bellman/expert_inv_Bellman', -expert_r_loss.item(), self.update_steps)
            self.add_scalar('inv_Bellman/rb_inv_Bellman', torch.mean(both_r[self.batch_size:]), self.update_steps)
            # record Q value estimation
            self.add_scalar('value_est/expert_Q', torch.mean(both_current_Q[:self.batch_size]), self.update_steps)
            self.add_scalar('value_est/rb_Q', torch.mean(both_current_Q[self.batch_size:]), self.update_steps)
            # record
            self.add_scalar('action_gap/expert_ag', torch.mean((both_current_Q - both_v)[:self.batch_size]), self.update_steps)
            self.add_scalar('action_gap/rb_ag', torch.mean((both_current_Q - both_v)[self.batch_size:]), self.update_steps)


    def grad_penalty(self, expert_obs_act, rb_obs_act, coff=1):
        alpha = torch.rand(expert_obs_act.size()[0], 1)
        alpha = alpha.expand_as(expert_obs_act).to(self.device)

        mix_obs_act = alpha * expert_obs_act + (1 - alpha) * rb_obs_act

        total_mix_obs_act = Variable(mix_obs_act, requires_grad=True)
        total_q = self.agent.get_value(total_mix_obs_act, both=self.agent.is_double_Q)[0]
        ones = torch.ones(total_q.size()).to(self.device)

        gradient = grad(
            outputs=total_q,
            inputs=total_mix_obs_act,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0] + EPS

        grad_penalty = coff * (gradient.norm(2, dim=1) - 1).pow(2).mean()
        return grad_penalty




















