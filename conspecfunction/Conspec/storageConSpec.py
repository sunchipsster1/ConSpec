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
        action_shape = 1
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
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
        self.successTake = 16
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
        self.r_batchS = torch.zeros(self.num_steps, self.success, 1)
        self.recurrent_hidden_statesS = torch.zeros(self.num_steps + 1, self.success, self.recurrent_hidden_state_size)
        self.act_batchS = torch.zeros(self.num_steps, self.success, action_shape)
        self.act_batchS = self.act_batchS.long()
        self.masks_batchS = torch.zeros(self.num_steps + 1, self.success, 1)
        self.stepS = 0
        self.obs_batchF = torch.zeros(self.num_steps + 1, self.success, *self.obs_shape)
        self.r_batchF = torch.zeros(self.num_steps, self.success, 1)
        self.recurrent_hidden_statesF = torch.zeros(self.num_steps + 1, self.success, self.recurrent_hidden_state_size)
        self.act_batchF = torch.zeros(self.num_steps, self.success, action_shape)
        self.act_batchF = self.act_batchF.long()
        self.masks_batchF = torch.zeros(self.num_steps + 1, self.success, 1)
        self.stepF = 0
        for i in range(self.heads):
            self.obs_batchheadsS[i] = torch.zeros(self.num_steps + 1, self.success, *self.obs_shape)
            self.r_batchheadsS[i] = torch.zeros(self.num_steps, self.success, 1)
            self.recurrent_hidden_statesbatchheadsS[i] = torch.zeros(self.num_steps + 1, self.success,
                                                                     self.recurrent_hidden_state_size)
            self.act_batchheadsS[i] = torch.zeros(self.num_steps, self.success, action_shape)
            self.act_batchheadsS[i] = self.act_batchheadsS[i].long()
            self.masks_batchheadsS[i] = torch.zeros(self.num_steps + 1, self.success, 1)
            self.obs_batchheadsF[i] = torch.zeros(self.num_steps + 1, self.success, *self.obs_shape)
            self.r_batchheadsF[i] = torch.zeros(self.num_steps, self.success, 1)
            self.recurrent_hidden_statesbatchheadsF[i] = torch.zeros(self.num_steps + 1, self.success,
                                                                     self.recurrent_hidden_state_size)
            self.act_batchheadsF[i] = torch.zeros(self.num_steps, self.success, action_shape)
            self.act_batchheadsF[i] = self.act_batchheadsF[i].long()
            self.masks_batchheadsF[i] = torch.zeros(self.num_steps + 1, self.success, 1)

        self.keysUsed = torch.zeros(head,)
        self.goodones = torch.zeros(head,)
    def contrastvalueReward(self, contrastval):
        self.rewardsORIG = torch.clone(self.rewards)
        self.rewards = self.rewards  + contrastval.unsqueeze(-1)
        return self.rewards



    def retrievekeysused(self):
        return self.keysUsed , self.goodones

    def storekeysused(self, keysUsed,goodones):  # this arises ONLY from the need to add the WHOLE SET, it is evoked ONCE.
        self.keysUsed=keysUsed
        self.goodones=goodones

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

    def addPosNeg(self, ForS, device):
        totalreward = self.rewards[-10:].sum(0)

        if ForS == 1:
            rewardssortgood = torch.nonzero(totalreward > 0.5).reshape(-1, )
            indicesrewardbatch = rewardssortgood[0::2]
            obsxx = self.obs[:, indicesrewardbatch].to(device)
            numberaddedxx = obsxx.shape[1]
            if numberaddedxx >1:
                indicesrewardbatch = rewardssortgood[0:4:2]
        else:
            rewardssortbad = torch.nonzero(totalreward < 0.5).reshape(-1, )
            indicesrewardbatch = rewardssortbad[0::2]
        obs = self.obs[:, indicesrewardbatch].to(device)
        rec = self.recurrent_hidden_states[:, indicesrewardbatch].to(device)
        masks = self.masks[:, indicesrewardbatch].to(device)
        act = self.actions[:, indicesrewardbatch].to(device)
        rew = self.rewards[:, indicesrewardbatch].to(device)
        numberadded = obs.shape[1]

        # '''
        if numberadded > 0:
            if ForS == 1:
                numcareabout = self.stepS
            elif ForS == 0:
                numcareabout = self.stepF
            if numberadded + numcareabout <= self.obs_batchS.shape[1]:
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
            #'''
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

            elif numcareabout == self.obs_batchS.shape[
                1]:
                hidden_state = rec
                masks = masks

                if ForS == 1:
                    lenconsider = obs.shape[1]
                    self.obs_batchS = torch.cat((self.obs_batchS , obs),1)
                    self.r_batchS = torch.cat((self.r_batchS , rew),1)
                    self.recurrent_hidden_statesS = torch.cat((self.recurrent_hidden_statesS , rec),1)
                    self.act_batchS = torch.cat((self.act_batchS , act),1)
                    self.masks_batchS = torch.cat((self.masks_batchS , masks),1)
                    self.obs_batchS = self.obs_batchS[:,lenconsider:]
                    self.r_batchS = self.r_batchS[:,lenconsider:]
                    self.recurrent_hidden_statesS = self.recurrent_hidden_statesS[:,lenconsider:]
                    self.act_batchS = self.act_batchS[:,lenconsider:]
                    self.masks_batchS = self.masks_batchS[:,lenconsider:]

                elif ForS == 0:
                    lenconsider =  obs.shape[1]
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
            self.obs_batchheadsS[i] = self.obs_batchheadsS[i].to(device)
            self.r_batchheadsS[i] = self.r_batchheadsS[i].to(device)
            self.recurrent_hidden_statesbatchheadsS[i] = self.recurrent_hidden_statesbatchheadsS[i].to(device)
            self.act_batchheadsS[i] = self.act_batchheadsS[i].to(device)
            self.masks_batchheadsS[i] = self.masks_batchheadsS[i].to(device)
            self.obs_batchheadsF[i] = self.obs_batchheadsF[i].to(device)
            self.r_batchheadsF[i] = self.r_batchheadsF[i].to(device)
            self.recurrent_hidden_statesbatchheadsF[i] = self.recurrent_hidden_statesbatchheadsF[i].to(device)
            self.act_batchheadsF[i] = self.act_batchheadsF[i].to(device)
            self.masks_batchheadsF[i] = self.masks_batchheadsF[i].to(device)

    def insert(self, obs, recurrent_hidden_states, actions,  rewards, masks, bad_masks):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step +
                                     1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.step = (self.step + 1) % self.num_steps

    def insertall(self, obs, recurrent_hidden_states, actions, rewards, masks, bad_masks):
        self.obs = obs
        self.recurrent_hidden_states = recurrent_hidden_states
        self.actions = actions
        self.rewards=rewards
        self.masks=masks
        self.bad_masks=bad_masks
        # self.step = (self.step + 1) % self.num_steps

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

    def releaseB(self):
        return self.obs, self.rewardsORIG, self.recurrent_hidden_states, self.actions, self.masks

    def releaseheadsSF(self, head):
        permS = torch.randperm(self.success)
        permF = torch.randperm(self.success)
        permS = permS[:self.successTake]
        permF = permF[:self.successTake]
        rew_batch = torch.cat((self.r_batchheadsS[head][:, permS], self.r_batchheadsF[head][:, permF]), dim=1)
        obs_batch = torch.cat((self.obs_batchheadsS[head][:, permS], self.obs_batchheadsF[head][:, permF]), dim=1)
        recurrent_hidden_states = torch.cat((self.recurrent_hidden_statesbatchheadsS[head][:, permS],
                                             self.recurrent_hidden_statesbatchheadsF[head][:, permF]), dim=1)
        act_batch = torch.cat((self.act_batchheadsS[head][:, permS], self.act_batchheadsF[head][:, permF]), dim=1)
        masks_batch = torch.cat((self.masks_batchheadsS[head][:, permS], self.masks_batchheadsF[head][:, permF]), dim=1)
        return obs_batch, rew_batch, recurrent_hidden_states, act_batch, masks_batch

    def releaseSF(self):
        permS = torch.randperm(self.success)
        permF = torch.randperm(self.success)
        permS = permS[:self.successTake]
        permF = permF[:self.successTake]

        rew_batch = torch.cat((self.r_batchS[:, permS], self.r_batchF[:, permF]), dim=1)
        obs_batch = torch.cat((self.obs_batchS[:, permS], self.obs_batchF[:, permF]), dim=1)
        recurrent_hidden_states = torch.cat(
            (self.recurrent_hidden_statesS[:, permS], self.recurrent_hidden_statesF[:, permF]), dim=1)
        act_batch = torch.cat((self.act_batchS[:, permS], self.act_batchF[:, permF]), dim=1)
        masks_batch = torch.cat((self.masks_batchS[:, permS], self.masks_batchF[:, permF]), dim=1)
        return obs_batch, rew_batch, recurrent_hidden_states, act_batch, masks_batch

    def feed_attnRB(self):
        obs_batchx, rew_batchx, recurrent_hidden_statesx, act_batchx, masks_batchx = self.releaseB()
        obs_batch = obs_batchx[:-1].view(-1, *self.obs.size()[2:])
        obs_batchorig = obs_batchx[:-1]
        recurrent_hidden_states_batch = recurrent_hidden_statesx[:-1].view(-1, self.recurrent_hidden_states.size(-1))
        actions_batch = act_batchx.view(-1, self.actions.size(-1))
        masks_batch = masks_batchx[:-1].view(-1, 1)
        reward_batch = rew_batchx.squeeze()  # [:-1]#.view(-1, 1)
        return obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, reward_batch

    def feed_attnRSF(self):
        obs_batchx, rew_batchx, recurrent_hidden_statesx, act_batchx, masks_batchx = self.releaseSF()
        obs_batch = obs_batchx[:-1].view(-1, *self.obs.size()[2:])
        obs_batchorig = obs_batchx[:-1]
        recurrent_hidden_states_batch = recurrent_hidden_statesx[:-1].view(-1, self.recurrent_hidden_states.size(-1))
        actions_batch = act_batchx.view(-1, self.actions.size(-1))
        masks_batch = masks_batchx[:-1].view(-1, 1)
        reward_batch = rew_batchx.squeeze()  # [:-1]#.view(-1, 1)
        return obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, reward_batch

    def feed_attnRSFheads(self, head):
        obs_batchx, rew_batchx, recurrent_hidden_statesx, act_batchx, masks_batchx = self.releaseheadsSF(head)
        obs_batch = obs_batchx[:-1].view(-1, *self.obs.size()[2:])
        obs_batchorig = obs_batchx[:-1]
        recurrent_hidden_states_batch = recurrent_hidden_statesx[:-1].view(-1, self.recurrent_hidden_states.size(-1))
        actions_batch = act_batchx.view(-1, self.actions.size(-1))
        masks_batch = masks_batchx[:-1].view(-1, 1)
        reward_batch = rew_batchx.squeeze()  # [:-1]#.view(-1, 1)
        return obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, reward_batch
