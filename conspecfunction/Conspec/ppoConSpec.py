import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPOConSpec():
    def __init__(self,
                 actor_criticCL,
                 module,
                 lrCL=None,
                 eps=None,
                 seed=None,
                factorR=None):
                 # actor_criticCL,
                 # clip_param,
                 # ppo_epoch,
                 # num_mini_batch,
                 # value_loss_coef,
                 # entropy_coef,
                 # module,
                 # choiceCLparams,
                 # args,
                 # lrCL=None,
                 # lr=None,
                 # eps=None,
                 # max_grad_norm=None,
                 # use_clipped_value_loss=True):

         self.moduleCL = module
         self.listparams = list(actor_criticCL.parameters()) + list(self.moduleCL.parameters())
         self.optimizerCL = optim.Adam(self.listparams, lr=lrCL, eps=eps)
         self.actor_criticCL = actor_criticCL
         self.seed = seed
         self.factorR = factorR

    def fictitiousReward(self,rollouts,keysUsed):
        #################Intrinsic reward
        obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, reward_batch = rollouts.feed_attnR()
        _, _, _, _, hidden = self.actor_criticCL.evaluate_actionsHiddens(
            obs_batch, recurrent_hidden_states_batch, masks_batch,
            actions_batch)
        hidden = hidden.view(*obs_batchorig.size()[:2],-1)
        _, attentionCLattn, _ = self.moduleCL(hidden, reward_batch, -1, obs_batchorig, 111,  self.seed,0)

        attentionCLattnnp = attentionCLattn.view(*obs_batchorig.size()[:2],-1)  # length, minibatch, keyhead
        goodz,_ = torch.max(attentionCLattnnp, dim=0)  # =  minibatch, keyhead
        size1, size2, size3 = attentionCLattnnp.size()
        greater = (( attentionCLattnnp > 0.6)) * attentionCLattnnp ####
        filterCorr = torch.tile(torch.reshape(keysUsed, (1,1, -1)), (size1, size2, 1))
        filterCorr1 = filterCorr  * self.factorR
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

        return rollouts.contrastvalueReward(sendContrastvalue)

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
                    costCL0, attentionCL0, ww = self.moduleCL(hidden, reward_batch, iii, obs_batchorig, iterate, self.seed, 1)
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
                    costCL0, attentionCL0, ww = self.moduleCL(hidden, reward_batch, iii, obs_batchorig,  iterate, self.seed,0)
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
            if self.factorR < 0.01:
                pass
            else:
                self.optimizerCL.zero_grad()
                costCL.backward()
                self.optimizerCL.step()
            print('trained!')


        return keysUsed, goodones
