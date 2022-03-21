import os
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import deque
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])

class ExpertStorage(object):
    def __init__(self, expert_file_path, env_name, start_seed, total_expert_trajs, num_trajs, batch_size,
                 device, obs_norm=True, absorbing=True, max_episode_length=None):

        self.env_name = env_name
        self.start_idx = start_seed
        self.total_expert_trajs = total_expert_trajs
        self.num_trajs = num_trajs
        self.expert_file_path = expert_file_path

        self.max_episode_length = max_episode_length
        self.obs_norm = obs_norm
        self.absorbing = absorbing

        self.batch_size = batch_size
        self.device = device

        self.shift, self.scale, self.std = None, None, None

        self.obs_acs_file_list = self.sample_trajs_file()

    def get_mean_var(self):
        assert self.obs_norm and self.shift is not None and self.std is not None, \
            'must choose the \'norm_obs\' option and implement \'load_transition_dataset()\' before getting statistics'
        return (self.shift, np.square(self.std))

    def sample_trajs_file(self):
        obs_acs_file_list = []
        obs_acs_file_path = '{}_obs_acs_traj={}.npz'
        select_idx = random.sample(list(range(1, self.total_expert_trajs+1)), self.num_trajs)
        for idx in select_idx:
            # idx = (i % self.total_expert_trajs) + 1
            obs_acs_file_list.append(obs_acs_file_path.format(self.env_name, idx))

        return obs_acs_file_list

    def load_trasition_dataset(self, infinite_bootstrap=False):
        """
                !!!obs & acs file should be corresponded!!!
        :param obs_file_path: obs file list includes the file names of reading npy
        :param acs_file_path: acs file list includes the file names of reading npy
        :return:
        """
        assert not (infinite_bootstrap and self.absorbing), 'infinite bootstrap and absorbing have similar effect!'

        obs_list, acs_list, next_obs_list, done_list = [], [], [], []
        for obs_acs_file in self.obs_acs_file_list:
            print('---- load obs and acs file:{} ----'.format(obs_acs_file))
            with open(os.path.join(self.expert_file_path, obs_acs_file), 'rb') as file:
                data = np.load(file)
                obs_arr = data['states'] # (len, dim)
                acs_arr = data['actions'] # (len, dim)

            assert obs_arr.shape[0] - 1 == acs_arr.shape[0], 'default one more terminal state should be stored in obs file'

            obs_list.append(obs_arr[:-1].copy()) # copy() is unnecessary, because elements in list will be concatenated later
            acs_list.append(acs_arr)
            next_obs_list.append(obs_arr[1:].copy())

            done_arr = np.zeros([obs_arr.shape[0]-1, ], dtype=np.float32)
            if infinite_bootstrap:# in this case, if traj terminates due to timelimit, the last done should be set as 0.
                if acs_arr.shape[0] < self.max_episode_length:
                    done_arr[-1] = 1.
            else:
                done_arr[-1] = 1.

            done_list.append(np.expand_dims(done_arr, axis=1))

        # (bs, dim)
        expert_obs = np.concatenate(obs_list, axis=0)
        expert_acs = np.concatenate(acs_list, axis=0)
        expert_next_obs = np.concatenate(next_obs_list, axis=0)
        # (bs, 1)
        expert_dones = np.concatenate(done_list, axis=0)

        assert expert_obs.shape[0] == expert_acs.shape[0] == expert_next_obs.shape[0] == expert_dones.shape[0], \
            'different length in expert data'

        print(f'##### loading total {expert_obs.shape[0]} expert samples #####')


        # obs & acs dim
        self.ori_obs_dim = expert_obs.shape[1:]
        self.ori_acs_dim = expert_acs.shape[1:]

        if self.obs_norm:
            assert len(self.ori_obs_dim) == 1, 'observation normalization only for mujoco envs'
            print('##### preprocess: observation normalization #####')
            self.shift = np.mean(expert_obs, 0)
            # todo: different with the ones in VecNormalize wrapper
            self.std = np.std(expert_obs, 0)
            self.scale = 1.0 / (self.std + 1e-3)
            expert_obs = (expert_obs - self.shift) * self.scale
            expert_next_obs = (expert_next_obs - self.shift) * self.scale

            print('expert samples statistics:')
            print('mean:{}'.format(self.shift))
            print('std:{}'.format(self.std))

        # todo: rescale storage action into the original unnormalized action space (-1, 1),
        # todo: in fact, only Humanoid action space (-0.4, 0.4) don't lie in (-1, 1)

        if self.absorbing: # have asserted that not (is_atari and absorbing)
            print('##### preprocess: absorbing state augment #####')
            expert_obs, expert_acs, expert_next_obs, expert_dones, augment_num = self.add_absorbing_state(
                expert_obs, expert_acs, expert_next_obs, expert_dones)
            print(f'total add {augment_num} absorbing states')


        expert_obs = torch.from_numpy(expert_obs).to(self.device)
        expert_acs = torch.from_numpy(expert_acs).to(self.device)
        expert_next_obs = torch.from_numpy(expert_next_obs).to(self.device)
        expert_dones = torch.from_numpy(expert_dones).to(self.device)

        expert_dataset = DataLoader(TensorDataset(expert_obs, expert_acs, expert_next_obs, expert_dones),
                                    batch_size=self.batch_size, shuffle=True, drop_last=True)

        return expert_dataset


    def add_absorbing_state(self, expert_obs, expert_acs, expert_next_obs, expert_dones):
        expert_obs = np.pad(expert_obs, ((0, 0), (0, 1)), mode='constant')
        expert_next_obs = np.pad(expert_next_obs, ((0, 0), (0, 1)), mode='constant')

        augment_num = 0
        i = 0
        current_len = 0
        while i < len(expert_obs): # len(self.expert_obs) is dynamically changing
            current_len += 1
            if expert_dones[i] == 1. and current_len < self.max_episode_length:
                current_len = 0
                expert_obs = np.insert(expert_obs, i+1, self.generate_absorbing_state(), axis=0)
                expert_next_obs[i] = self.generate_absorbing_state()
                expert_next_obs = np.insert(expert_next_obs, i+1, self.generate_absorbing_state(), axis=0)
                expert_acs = np.insert(expert_acs, i+1, self.generate_dummy_action(), axis=0)
                expert_dones[i] = 0.0
                expert_dones = np.insert(expert_dones, i+1, np.array(1.0), axis=0)
                i += 1
                augment_num += 1
            i += 1

        return expert_obs, expert_acs, expert_next_obs, expert_dones, augment_num

    def generate_absorbing_state(self):
        absorbing_state = np.zeros((self.ori_obs_dim[0] + 1, ), dtype=np.float32)
        absorbing_state[-1] = 1.
        return absorbing_state

    def generate_dummy_action(self):
        dummy_action = np.zeros((self.ori_acs_dim[0], ), dtype=np.float32)
        return dummy_action


class ReplayBuffer(object):
    def __init__(self, capacity, obs_shape, action_space, device):
        """
        todo: multiprocessing storage
        :param capacity: the max number of samples stored in the replay buffer
        :param device:
        """

        # self.capacity = capacity
        # self.device = device
        # self.total_num = 0
        #
        # self.obs = deque(maxlen=capacity + 1)
        # self.obs.append(None) # for the substitution in add_batch()
        # self.next_obs = deque(maxlen=capacity)
        # self.acs = deque(maxlen=capacity)
        # self.dones = deque(maxlen=capacity)


        # self.capacity = capacity
        # self.device = device
        # self.obs = torch.zeros(capacity + 1, *obs_shape).to(device)
        #
        # self.action_shape = action_space.shape[0]
        # self.actions = torch.zeros(capacity + 1, self.action_shape).to(self.device)
        #
        # self.offset = 0
        # self.steps = 0


        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros(capacity + 1, *obs_shape).to(device)
        self.action_shape = action_space.shape[0]
        self.actions = torch.zeros(capacity + 1, self.action_shape).to(device)
        self.dones = torch.zeros(capacity + 1,).to(device)
        self.next_obs = [None] * (self.capacity+1)

        self.offset = 0
        self.steps = 0





    def add_batch(self, obs_tensor, acs_tensor, next_obs_tensor, done, truncated_done):
        """
            note: adopt the inifite bootstrap trick in 'iq-learn', i.e. done is False when traj is over due to timelimit
        :param obs_tensor: shape: (dim,)
        :param acs_tensor: shape: (dim,)
        :param next_obs_tensor: (dim,)
        :param done: False or True, real return by the env.step()
        :param truncated_done: True (done must be True) or False
        :return:
        """

        # # substitute the current obs
        # self.obs[-1] = obs_tensor
        #
        # # append the next obs
        # if done:
        #     self.obs.append(None) # add for next substitution
        #     self.next_obs.append(next_obs_tensor)
        #     if truncated_done:
        #         self.dones.append([0.])
        #     else:
        #         self.dones.append([1.])
        # else:
        #     self.obs.append(next_obs_tensor)
        #     self.next_obs.append(None) # add an empty next obs
        #     self.dones.append([0.])
        #
        # self.acs.append(acs_tensor)
        #
        # self.total_num = min(self.total_num+1, self.capacity)



        # idx = self.steps % (self.capacity + 1)
        # next_idx = (self.steps + 1) % (self.capacity + 1)
        #
        #
        # self.obs[idx].copy_(obs_tensor)
        # self.actions[idx].copy_(acs_tensor)
        # self.obs[next_idx].copy_(next_obs_tensor)
        #
        #
        # self.steps += 1
        # if self.steps > self.capacity:
        #     self.offset = (self.offset + 1 )%(self.capacity + 1)



        idx = self.steps % (self.capacity + 1)
        next_idx = (self.steps + 1) % (self.capacity + 1)

        self.obs[idx].copy_(obs_tensor)
        self.actions[idx].copy_(acs_tensor)
        if done:
            self.next_obs[idx] = next_obs_tensor
            self.dones[idx] = 0. if truncated_done else 1.
        else:
            self.next_obs[idx] = None
            self.obs[next_idx].copy_(next_obs_tensor)
            self.dones[idx] = 0.

        self.steps += 1
        if self.steps > self.capacity:
            self.offset = (self.offset + 1) % (self.capacity + 1)


    def sample(self, batch_size):
        """
        :param batch_size:
        :return:
        """


        # batch_idx = random.sample(list(range(0, self.total_num)), batch_size)
        # batch_obs, batch_acs, batch_next_obs, batch_dones = [], [], [], []
        # for idx in batch_idx:
        #     batch_obs.append(self.obs[idx])
        #     batch_acs.append(self.acs[idx])
        #     batch_dones.append(self.dones[idx])
        #
        #     next_obs = self.next_obs[idx]
        #     if next_obs is not None:
        #         batch_next_obs.append(next_obs)
        #     else:
        #         batch_next_obs.append(self.obs[idx+1])
        #
        # batch_obs = torch.stack(batch_obs, dim=0).to(device=self.device) # (bs, dim)
        # batch_acs = torch.stack(batch_acs, dim=0).to(device=self.device) # (bs, dim)
        # batch_next_obs = torch.stack(batch_next_obs, dim=0).to(device=self.device)# (bs, dim)
        # batch_dones = torch.tensor(batch_dones).to(device=self.device) # (bs, 1)


        # base_idx = random.sample(list(range(0, min(self.capacity, self.steps))), batch_size)
        # base_idx = np.array(base_idx)
        # batch_idx = torch.tensor((base_idx + self.offset) % (self.capacity + 1)).to(self.device)
        # batch_next_idx = torch.tensor((base_idx + self.offset + 1) % (self.capacity + 1)).to(self.device)
        #
        # batch_obs = torch.index_select(self.obs, dim=0, index=batch_idx)
        # batch_acs = torch.index_select(self.actions, dim=0, index=batch_idx)
        # batch_next_obs = torch.index_select(self.obs, dim=0, index=batch_next_idx)
        # batch_dones = []



        base_idx = random.sample(list(range(0, min(self.capacity, self.steps))), batch_size)
        base_idx = np.array(base_idx)

        np_batch_idx = (base_idx + self.offset) % (self.capacity + 1)
        np_batch_next_idx = (base_idx + self.offset + 1) % (self.capacity + 1)

        batch_idx = torch.tensor(np_batch_idx).to(self.device)
        batch_next_idx = torch.tensor(np_batch_next_idx).to(self.device)


        batch_obs = torch.index_select(self.obs, dim=0, index=batch_idx)
        batch_acs = torch.index_select(self.actions, dim=0, index=batch_idx)
        batch_next_obs = torch.index_select(self.obs, dim=0, index=batch_next_idx)
        batch_dones = torch.index_select(self.dones, dim=0, index=batch_idx)

        for i, idx in enumerate(np_batch_idx):
            next_obs = self.next_obs[idx]
            if next_obs is not None:
                batch_next_obs[i].copy_(next_obs)

        return batch_obs, batch_acs, batch_next_obs, batch_dones


    def compute_value(self):
        # todo: for RL algorithm which should compute the target value
        raise NotImplementedError


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step +
                                     1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ


