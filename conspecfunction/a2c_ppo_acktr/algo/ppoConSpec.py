import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPOConSpec():
    def __init__(self,
                 actor_critic,actor_criticCL,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 module,
                 choiceCLparams,
                 args,
                 lrCL=None,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.actor_critic = actor_critic
        self.actor_criticCL = actor_criticCL

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.moduleCL = module
        self.args = args
        self.listparams = list(actor_criticCL.parameters()) + list(self.moduleCL.parameters())
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.optimizerCL = optim.Adam(self.listparams, lr=lrCL, eps=eps)

    def fictitiousReward(self,rollouts,keysUsed,device,iteration):
        #################Intrinsic reward
        obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, reward_batch = rollouts.feed_attnR()
        _, _, _, _, hidden = self.actor_criticCL.evaluate_actionsHiddens(
            obs_batch, recurrent_hidden_states_batch, masks_batch,
            actions_batch)
        hidden = hidden.view(*obs_batchorig.size()[:2],-1)
        _, attentionCLattn, _ = self.moduleCL(hidden, reward_batch, -1, obs_batchorig, 111,  self.args.seed,0)

        attentionCLattnnp = attentionCLattn.view(*obs_batchorig.size()[:2],-1)  # length, minibatch, keyhead
        goodz,_ = torch.max(attentionCLattnnp, dim=0)  # =  minibatch, keyhead
        size1, size2, size3 = attentionCLattnnp.size()
        greater = (( attentionCLattnnp > 0.6)) * attentionCLattnnp ####
        filterCorr = torch.tile(torch.reshape(keysUsed, (1,1, -1)), (size1, size2, 1))
        filterCorr1 = filterCorr  * self.args.factorR
        sendContrastvalue = (greater * filterCorr1)
        #'''
        roundhalf = 3
        round = roundhalf
        allvalues = []
        for orthoit in range(round * 2 + 1):
            temp = torch.roll(sendContrastvalue, orthoit - roundhalf, dims=0)
            if orthoit - roundhalf > 0:
                temp[:(orthoit - roundhalf)] = 0.
            allvalues.append(temp)  # 80,16,10 heads
        allvalues = torch.stack(allvalues, dim=0)
        allvaluesmax,_ = torch.max(allvalues, dim=0)
        allvaluesdifference = sendContrastvalue - allvaluesmax
        sendContrastvalue[allvaluesdifference < 0.] = 0.
        sendContrastvalue[sendContrastvalue < 0.] = 0.
        sendContrastvaluesummed = 0.
        for orthoit in range(round):
            temp = torch.roll(sendContrastvalue, orthoit + 1, dims=0) * (.5 ** (orthoit))
            temp[:orthoit] = 0.
            sendContrastvaluesummed += temp
        sendContrastvaluesummed = sendContrastvaluesummed
        sendContrastvalue = sendContrastvalue - sendContrastvaluesummed
        sendContrastvalue[sendContrastvalue < 0.] = 0.
        sendContrastvalue = sendContrastvalue.sum(2)
        #'''
        print('intrinsic R!')

        rollouts.contrastvalueReward(sendContrastvalue)

    def update(self, rollouts, head, keysUsed,goodones,iterate):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        wwtotalpos = []
        wwtotalneg = []
        attentionCL = []
        costCL = 0
        #################ConSpec module is learned
        if rollouts.stepS > rollouts.success -1:

            ########################
            for iii in range(head):

                #'''
                if keysUsed[iii] > 0.5:
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, reward_batch = rollouts.feed_attnRSFheads(iii)
                    _, _, _, _, hidden = self.actor_criticCL.evaluate_actionsHiddens(
                        obs_batch, recurrent_hidden_states_batch, masks_batch,
                        actions_batch)
                    hidden = hidden.view(*obs_batchorig.size()[:2], -1)
                    costCL0, attentionCL0, ww = self.moduleCL(hidden, reward_batch, iii, obs_batchorig, iterate, self.args.seed, 1)
                    attentionCL.append(attentionCL0[:,:,iii].squeeze().transpose(1,0))
                    costCL += costCL0
                    wwtotalpos.append(ww[0][iii].detach().cpu())
                    wwtotalneg.append(ww[1][iii].detach().cpu())
                else:
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, reward_batch = rollouts.feed_attnRSF()

                    _, _, _, _, hidden = self.actor_criticCL.evaluate_actionsHiddens(
                        obs_batch, recurrent_hidden_states_batch, masks_batch,
                        actions_batch)
                    hidden = hidden.view(*obs_batchorig.size()[:2], -1)
                    costCL0, attentionCL0, ww = self.moduleCL(hidden, reward_batch, iii, obs_batchorig,  iterate, self.args.seed,0)
                    cossimtotalmaxxx, _ = (torch.max(attentionCL0, dim=0))
                    attentionCL.append(attentionCL0[:,:,iii].squeeze().transpose(1,0))
                    costCL += costCL0
                    wwtotalpos.append(ww[0][iii].detach().cpu())
                    wwtotalneg.append(ww[1][iii].detach().cpu())
                #'''

            for indall in range(head):
                if (wwtotalpos[indall] - wwtotalneg[indall] > 0.6) and wwtotalpos[indall] > 0.6:
                    goodones[indall] += 1
                else:
                    goodones[indall] = 0
            for iii in range(head):
                if goodones[iii] > 25 and keysUsed[iii] < 0.1:
                    keysUsed[iii] = 1.
                    rollouts.storeheadsSF(iii)
            if self.args.factorR < 0.01:
                pass
            else:
                self.optimizerCL.zero_grad()
                costCL.backward()
                self.optimizerCL.step()
            print('trained!')

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)


                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()
                #'''

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef ).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()
                #'''

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, keysUsed, goodones
