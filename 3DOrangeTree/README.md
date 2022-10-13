


# ConSpec on the OrangeTree task

## Information
The 3D environments were built in SilicoLabs Experimenter software (https://www.silicolabs.ca/), which is based on the Unity framework. 

This code is built on top of the basic PPO code taken from the repository: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

The interface between the environment and Pytorch was built on top of ML agents: https://github.com/Unity-Technologies/ml-agents


## Installation
Make sure  to unzip the 3 environment binary files that are contained in this subfolder. 
To create this environment, install the packages in the requirements.txt file
In main.py --> modify the directory line 52. The directory line should where you placed the environment files. 


### Launching

#### To run ConSpec on the OrangeTree task: 
In the command line, first access the environment by running: 
```
chmod -R 755 /home/chen/PycharmProjects/ProjTreeSimplePink2extrahardLinux4590_338084842miss/GridWorld.x86_64
```
Then run: 
```
python main.py  --algo Conspec  --use-gae --lr 2e-4 --clip-param 0.08 --value-loss-coef 0.5 --num-processes 16 --num-steps 65 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.02 --lrCL 20e-4 --choiceCLparams 0 --seed 80000  --head 8 --factorR 0.5
```
#### To test the runs from main.py on the modified OrangeTree task with novel black objects: 
In the command line, first access the environment by running: 
```
chmod -R 755 /home/chen/PycharmProjects/ProjTreeSimplePink2extrahardLinux4590_33_84842onevarb/GridWorld.x86_64 
```
Then run: 
```
python maintestBlackobj.py  --algo Conspec  --use-gae --lr 2e-4 --clip-param 0.08 --value-loss-coef 0.5 --num-processes 16 --num-steps 65 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.02  --lrCL 20e-4 --choiceCLparams 0 --seed 80000  --head 8 --factorR 0.5
```
#### To test the runs from main.py on the new green and pink environment:
In the command line, first access the environment by running: 
```
chmod -R 755 /home/chen/PycharmProjects/ProjTreeSimplePink2extrahardLinux4590_338084842TestmissRoommgp/GridWorld.x86_64  
```
Then run: 
```
python maintestNewContext.py  --algo Conspec  --use-gae --lr 2e-4 --clip-param 0.08 --value-loss-coef 0.5 --num-processes 16 --num-steps 65 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.02 --lrCL 20e-4 --choiceCLparams 0 --seed 80000  --head 8 --factorR 0.5
```


