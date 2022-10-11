import numpy
import torch
import torch.nn.functional as F
import torch.nn as nn



from Conspec.storageConSpec import RolloutStorage
from Conspec.ppoConSpec import PPOConSpec



class conspecfunctionx(nn.Module):
    def __init__(self, acmodelCL, moduleuse, head,num_processes,lrCL, obsspace,  num_actions, recurrent_hidden_state_size, num_steps,eps,seed,factorR):
        super(conspecfunctionx, self).__init__()

        self.num_procs = num_processes
        self.rollouts = RolloutStorage(num_steps, self.num_procs,obsspace, num_actions,
                              recurrent_hidden_state_size, head)  # envs.observation_space.shape
        self.acmodelCL = acmodelCL
        self.moduleuse = moduleuse
        self.head = head
        self.agent = PPOConSpec(
            self.acmodelCL,
            self.moduleuse,
            lrCL=lrCL,
            eps=eps,
            seed=seed,
            factorR=factorR)
        ##############################

    def storeandupdate(self,imagetodo, memorytodo, actiontodo, rewardtodo, maskstodo,iteration ):  # obs =  [600, 16, 1, 54, 64
        with torch.no_grad():

            self.rollouts.insertall(imagetodo, memorytodo, actiontodo, rewardtodo, maskstodo, maskstodo)  #
            device = torch.device("cuda")
            self.rollouts.to(device)
            self.rollouts.addPosNeg(1, device)
            self.rollouts.addPosNeg(0, device)

        keysUsed, goodones = self.rollouts.retrievekeysused()
        keysUsed, goodones = self.agent.update(self.rollouts, self.head, keysUsed.detach(), goodones.detach(),
                                               iteration)  # value_loss, action_loss, dist_entropy,
        keysUsedt = keysUsed.to(device=device)
        with torch.no_grad():

            rewardtotalintrisic = self.agent.fictitiousReward(self.rollouts, keysUsedt)
            self.rollouts.storekeysused(keysUsed, goodones)
        return rewardtotalintrisic



