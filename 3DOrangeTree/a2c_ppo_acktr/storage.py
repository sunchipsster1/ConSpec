import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size, head):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.num_processes = num_processes

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.rewardsORIG = torch.zeros(num_steps, num_processes, 1)

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
        self.obs_shape = obs_shape
        self.step = 0

        self.recurrent_hidden_state_size = recurrent_hidden_state_size
        self.heads = head 
        self.success = 16
        self.successTake = 8
        self.hidden_state_size = 256

        self.obs_batchheadsS = [None] * self.heads
        self.r_batchheadsS = [None] * self.heads
        self.recurrent_hidden_statesbatchheadsS = [None] * self.heads
        self.act_batchheadsS = [None] * self.heads
        self.masks_batchheadsS = [None] * self.heads
        self.stepheadsS = 0

        self.obs_batchheadsF = [None] * self.heads

        self.r_batchheadsF = [None] * self.heads
        self.recurrent_hidden_statesbatchheadsF = [None] * self.heads
        self.act_batchheadsF = [None] * self.heads
        self.masks_batchheadsF = [None] * self.heads
        self.stepheadsF = 0

        self.obs_batchS = torch.zeros(self.num_steps + 1, self.success, *self.obs_shape)
        self.r_batchS = torch.zeros(self.num_steps, self.success,1)
        self.recurrent_hidden_statesS = torch.zeros(self.num_steps + 1, self.success, self.recurrent_hidden_state_size)
        self.act_batchS = torch.zeros(self.num_steps, self.success, action_shape)
        self.act_batchS = self.act_batchS.long()
        self.masks_batchS = torch.zeros(self.num_steps+ 1, self.success, 1)
        self.stepS = 0
        self.obs_batchF = torch.zeros(self.num_steps + 1, self.success, *self.obs_shape)
        self.r_batchF = torch.zeros(self.num_steps, self.success,1)
        self.recurrent_hidden_statesF = torch.zeros(self.num_steps + 1, self.success, self.recurrent_hidden_state_size)
        self.act_batchF = torch.zeros(self.num_steps, self.success, action_shape)
        self.act_batchF = self.act_batchF.long()
        self.masks_batchF = torch.zeros(self.num_steps+ 1, self.success, 1)
        self.stepF = 0
        for i in range(self.heads):
            self.obs_batchheadsS[i] = torch.zeros(self.num_steps + 1, self.success, *self.obs_shape)
            self.r_batchheadsS[i] = torch.zeros(self.num_steps, self.success,1)
            self.recurrent_hidden_statesbatchheadsS[i] = torch.zeros(self.num_steps + 1, self.success, self.recurrent_hidden_state_size)
            self.act_batchheadsS[i] = torch.zeros(self.num_steps, self.success, action_shape)
            self.act_batchheadsS[i] = self.act_batchheadsS[i].long()
            self.masks_batchheadsS[i] = torch.zeros(self.num_steps+ 1, self.success, 1)
            self.obs_batchheadsF[i] = torch.zeros(self.num_steps + 1, self.success, *self.obs_shape)
            self.r_batchheadsF[i] = torch.zeros(self.num_steps, self.success,1)
            self.recurrent_hidden_statesbatchheadsF[i] = torch.zeros(self.num_steps + 1, self.success, self.recurrent_hidden_state_size)
            self.act_batchheadsF[i] = torch.zeros(self.num_steps, self.success, action_shape)
            self.act_batchheadsF[i] = self.act_batchheadsF[i].long()
            self.masks_batchheadsF[i] = torch.zeros(self.num_steps+ 1, self.success, 1)

    def contrastvalueReward(self, contrastval):
        self.rewardsORIG = torch.clone(self.rewards)
        self.rewards =  self.rewards + contrastval.unsqueeze(-1)

    def releaseB(self):
        return self.obs, self.rewardsORIG, self.recurrent_hidden_states, self.actions, self.masks
    def feed_attnRB(self):
        obs_batchx, rew_batchx, recurrent_hidden_statesx, act_batchx, masks_batchx = self.releaseB()
        obs_batch = obs_batchx[:-1].view(-1, *self.obs.size()[2:])
        obs_batchorig = obs_batchx[:-1]
        recurrent_hidden_states_batch = recurrent_hidden_statesx[:-1].view(-1, self.recurrent_hidden_states.size(-1))
        actions_batch = act_batchx.view(-1, self.actions.size(-1))
        masks_batch = masks_batchx[:-1].view(-1, 1)
        reward_batch = rew_batchx.squeeze() 
        return obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, reward_batch

    def retrievestepS(self):
        return self.stepS
    def retrieveR(self):
        return self.rewards
    def retrieveobs(self):
        return self.obs
    def retrieveeverything(self):
        return torch.cat((self.obs_batchS, self.obs_batchF), dim=1), torch.cat((self.r_batchS, self.r_batchF),
                                                                               dim=1), torch.cat(
            (self.masks_batchS, self.masks_batchF), dim=1), torch.cat((self.act_batchS, self.act_batchF), dim=1)
    def retrieveRS(self):
        return self.r_batchS, self.r_batchF


    def storeheadsSF(self, head): 
        self.obs_batchheadsS[head] = self.obs_batchS
        self.r_batchheadsS[head] = self.r_batchS
        self.recurrent_hidden_statesbatchheadsS[head] = self.recurrent_hidden_statesS
        self.act_batchheadsS[head] = self.act_batchS
        self.masks_batchheadsS[head] = self.masks_batchS
        self.obs_batchheadsF[head] = self.obs_batchF
        self.r_batchheadsF[head] = self.r_batchF
        self.recurrent_hidden_statesbatchheadsF[head] = self.recurrent_hidden_statesF
        self.act_batchheadsF[head] = self.act_batchF
        self.masks_batchheadsF[head] = self.masks_batchF

    def addPosNeg(self,ForS,device, args): 
        totalreward = self.rewards[-20:].sum(0)
        if ForS == 1:
            rewardssortgood = torch.nonzero(totalreward > 0.5).reshape(-1, )
            indicesrewardbatch = rewardssortgood[
                                 0::2] 
            obsxx = self.obs[:, indicesrewardbatch].to(device)
            numberaddedxx = obsxx.shape[1] 
            if numberaddedxx > 1:
                indicesrewardbatch = rewardssortgood[0:4:2]
        else:
            rewardssortbad = torch.nonzero(totalreward < 0.5).reshape(-1, )
            indicesrewardbatch = rewardssortbad[ 0::2]  
        obs = self.obs[:, indicesrewardbatch].to(device)
        rec = self.recurrent_hidden_states[:, indicesrewardbatch].to(device)
        masks = self.masks[:, indicesrewardbatch].to(device)
        act = self.actions[:, indicesrewardbatch].to(device)
        rew = self.rewards[:, indicesrewardbatch].to(device)
        numberadded = obs.shape[1]  # number of success obs to be added
        
        totalreward = self.r_batchS.sum(0)
        totalreward = self.r_batchF.sum(0)
        # '''
        if numberadded > 0:
            if ForS == 1:
                numcareabout = self.stepS
            elif ForS == 0:
                numcareabout = self.stepF
            if numberadded + numcareabout <= self.obs_batchS.shape[1]:  # i.e. add all the new obs
                if ForS == 1:
                    self.obs_batchS[:, self.stepS:self.stepS + numberadded] = obs
                    self.r_batchS[:, self.stepS:self.stepS + numberadded] = rew
                    self.recurrent_hidden_statesS[:, self.stepS:self.stepS + numberadded] = rec
                    self.act_batchS[:, self.stepS:self.stepS + numberadded] = act
                    self.masks_batchS[:, self.stepS:self.stepS + numberadded] = masks
                    self.stepS = (self.stepS + numberadded)
                elif ForS == 0:
                    self.obs_batchF[:, self.stepF:self.stepF + numberadded] = obs
                    self.r_batchF[:, self.stepF:self.stepF + numberadded] = rew
                    self.recurrent_hidden_statesF[:, self.stepF:self.stepF + numberadded] = rec
                    self.act_batchF[:, self.stepF:self.stepF + numberadded] = act
                    self.masks_batchF[:, self.stepF:self.stepF + numberadded] = masks
                    self.stepF = (self.stepF + numberadded)
            # '''
            elif (numberadded + numcareabout >= self.obs_batchS.shape[1]) and (
                    numcareabout < self.obs_batchS.shape[1]):  

                if ForS == 1:
                    numbertoadd = self.obs_batchS.shape[1] - self.stepS
                    self.obs_batchS[:, self.stepS:self.stepS + numbertoadd, :] = obs[:, :numbertoadd]
                    self.r_batchS[:, self.stepS:self.stepS + numbertoadd] = rew[:, :numbertoadd]
                    self.recurrent_hidden_statesS[:, self.stepS:self.stepS + numbertoadd] = rec[:, :numbertoadd]
                    self.act_batchS[:, self.stepS:self.stepS + numbertoadd] = act[:, :numbertoadd]
                    self.masks_batchS[:, self.stepS:self.stepS + numbertoadd] = masks[:, :numbertoadd]
                    self.stepS = (self.stepS + numbertoadd)
                elif ForS == 0:
                    numbertoadd = self.obs_batchS.shape[1] - self.stepF
                    self.obs_batchF[:, self.stepF:self.stepF + numbertoadd, :] = obs[:, :numbertoadd]
                    self.r_batchF[:, self.stepF:self.stepF + numbertoadd] = rew[:, :numbertoadd]
                    self.recurrent_hidden_statesF[:, self.stepF:self.stepF + numbertoadd] = rec[:, :numbertoadd]
                    self.act_batchF[:, self.stepF:self.stepF + numbertoadd] = act[:, :numbertoadd]
                    self.masks_batchF[:, self.stepF:self.stepF + numbertoadd] = masks[:, :numbertoadd]
                    self.stepF = (self.stepF + numbertoadd)

            elif numcareabout == self.obs_batchS.shape[1]:  
                masks = masks

                if ForS == 1:
                    lenconsider = obs.shape[1]
                    self.obs_batchS = torch.cat((self.obs_batchS, obs),
                                                1)  
                    self.r_batchS = torch.cat((self.r_batchS, rew),
                                              1)  
                    self.recurrent_hidden_statesS = torch.cat((self.recurrent_hidden_statesS, rec),
                                                              1)  
                    self.act_batchS = torch.cat((self.act_batchS, act),
                                                1)  
                    self.masks_batchS = torch.cat((self.masks_batchS, masks),
                                                  1)  
                    self.obs_batchS = self.obs_batchS[:, lenconsider:]
                    self.r_batchS = self.r_batchS[:, lenconsider:]
                    self.recurrent_hidden_statesS = self.recurrent_hidden_statesS[:, lenconsider:]
                    self.act_batchS = self.act_batchS[:, lenconsider:]
                    self.masks_batchS = self.masks_batchS[:, lenconsider:]

                elif ForS == 0:
                    lenconsider = obs.shape[1]
                    self.obs_batchF = torch.cat((self.obs_batchF, obs),
                                                1)  
                    self.r_batchF = torch.cat((self.r_batchF, rew),
                                              1)  
                    self.recurrent_hidden_statesF = torch.cat((self.recurrent_hidden_statesF, rec),
                                                              1)  
                    self.act_batchF = torch.cat((self.act_batchF, act),
                                                1)  
                    self.masks_batchF = torch.cat((self.masks_batchF, masks),
                                                  1)  
                    self.obs_batchF = self.obs_batchF[:, lenconsider:]
                    self.r_batchF = self.r_batchF[:, lenconsider:]
                    self.recurrent_hidden_statesF = self.recurrent_hidden_statesF[:, lenconsider:]
                    self.act_batchF = self.act_batchF[:, lenconsider:]
                    self.masks_batchF = self.masks_batchF[:, lenconsider:]


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

        self.obs_batchS = self.obs_batchS.to(device)
        self.r_batchS = self.r_batchS.to(device)
        self.recurrent_hidden_statesS = self.recurrent_hidden_statesS.to(device)
        self.act_batchS = self.act_batchS.to(device)
        self.masks_batchS = self.masks_batchS.to(device)
        self.obs_batchF = self.obs_batchF.to(device)
        self.r_batchF = self.r_batchF.to(device)
        self.recurrent_hidden_statesF = self.recurrent_hidden_statesF.to(device)
        self.act_batchF = self.act_batchF.to(device)
        self.masks_batchF = self.masks_batchF.to(device)
        for i in range(self.heads):
            self.obs_batchheadsS[i] =self.obs_batchheadsS[i].to(device)
            self.r_batchheadsS[i] =self.r_batchheadsS[i].to(device)
            self.recurrent_hidden_statesbatchheadsS[i] = self.recurrent_hidden_statesbatchheadsS[i].to(device)
            self.act_batchheadsS[i] =self.act_batchheadsS[i].to(device)
            self.masks_batchheadsS[i] =  self.masks_batchheadsS[i].to(device)
            self.obs_batchheadsF[i] = self.obs_batchheadsF[i].to(device)
            self.r_batchheadsF[i] = self.r_batchheadsF[i].to(device)
            self.recurrent_hidden_statesbatchheadsF[i] = self.recurrent_hidden_statesbatchheadsF[i].to(device)
            self.act_batchheadsF[i] = self.act_batchheadsF[i].to(device)
            self.masks_batchheadsF[i] = self.masks_batchheadsF[i].to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks, miniN):
        self.obs[self.step + 1, miniN].copy_(obs)
        self.recurrent_hidden_states[self.step +
                                     1, miniN].copy_(recurrent_hidden_states)
        self.actions[self.step, miniN].copy_(actions)
        self.action_log_probs[self.step, miniN].copy_(action_log_probs)
        self.value_preds[self.step, miniN].copy_(value_preds)
        self.rewards[self.step, miniN].copy_(rewards)
        self.masks[self.step + 1, miniN].copy_(masks)
        self.bad_masks[self.step + 1, miniN].copy_(bad_masks)
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

    def feed_attnR(self):
        obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])
        obs_batchorig = self.obs[:-1]
        recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
            -1, self.recurrent_hidden_states.size(-1))
        actions_batch = self.actions.view(-1,
                                          self.actions.size(-1))
        masks_batch = self.masks[:-1].view(-1, 1)
        reward_batch = self.rewards.squeeze()

        return obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, reward_batch
    def releaseheadsSF(self, head):
        permS = torch.randperm(self.success)
        permF = torch.randperm(self.success)
        permS = permS[:self.successTake]
        permF = permF[:self.successTake]
        rew_batch = torch.cat((self.r_batchheadsS[head][:, permS],self.r_batchheadsF[head][:, permF]),dim=1)
        obs_batch = torch.cat((self.obs_batchheadsS[head][:, permS],self.obs_batchheadsF[head][:, permF]),dim=1)
        recurrent_hidden_states=torch.cat((self.recurrent_hidden_statesbatchheadsS[head][:, permS],self.recurrent_hidden_statesbatchheadsF[head][:, permF]),dim=1)
        act_batch=torch.cat((self.act_batchheadsS[head][:, permS],self.act_batchheadsF[head][:, permF]),dim=1)
        masks_batch=torch.cat((self.masks_batchheadsS[head][:, permS],self.masks_batchheadsF[head][:, permF]),dim=1)
        return obs_batch, rew_batch,recurrent_hidden_states, act_batch, masks_batch

    def releaseSF(self):
        permS = torch.randperm(self.success)
        permF = torch.randperm(self.success)
        permS = permS[:self.successTake]
        permF = permF[:self.successTake]
        #print(permS)
        rew_batch = torch.cat((self.r_batchS[:, permS],self.r_batchF[:, permF]),dim=1)

        totalreward = rew_batch.sum(0)

        obs_batch = torch.cat((self.obs_batchS[:, permS],self.obs_batchF[:, permF]),dim=1)
        recurrent_hidden_states=torch.cat((self.recurrent_hidden_statesS[:, permS],self.recurrent_hidden_statesF[:, permF]),dim=1)
        act_batch=torch.cat((self.act_batchS[:, permS],self.act_batchF[:, permF]),dim=1)
        masks_batch=torch.cat((self.masks_batchS[:, permS],self.masks_batchF[:, permF]),dim=1)
        return obs_batch, rew_batch,recurrent_hidden_states, act_batch, masks_batch

    def feed_attnRSF(self):
        obs_batchx, rew_batchx, recurrent_hidden_statesx, act_batchx, masks_batchx = self.releaseSF()
        obs_batch = obs_batchx[:-1].view(-1, *self.obs.size()[2:])
        obs_batchorig = obs_batchx[:-1]
        recurrent_hidden_states_batch = recurrent_hidden_statesx[:-1].view( -1, self.recurrent_hidden_states.size(-1))
        actions_batch = act_batchx.view(-1,self.actions.size(-1))
        masks_batch = masks_batchx[:-1].view(-1, 1)
        reward_batch = rew_batchx.squeeze()
        return obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, reward_batch
    def feed_attnRSFheads(self,head):
        obs_batchx, rew_batchx, recurrent_hidden_statesx, act_batchx, masks_batchx = self.releaseheadsSF(head)
        obs_batch = obs_batchx[:-1].view(-1, *self.obs.size()[2:])
        obs_batchorig = obs_batchx[:-1]
        recurrent_hidden_states_batch = recurrent_hidden_statesx[:-1].view(-1, self.recurrent_hidden_states.size(-1))
        actions_batch = act_batchx.view(-1, self.actions.size(-1))
        masks_batch = masks_batchx[:-1].view(-1, 1)
        reward_batch = rew_batchx.squeeze()  
        return obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, reward_batch

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
