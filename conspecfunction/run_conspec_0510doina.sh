#!/bin/bash

lr=$1
lrCL=$2
seed=$3
factorR=$4
expansion=$5
factorC=$6

#python -u mainCLdelaysep84TVTregress.py  --algo ppoCLsep84  --factorC $factorC --use-gae --lr $lr --clip-param 0.08 --value-loss-coef 0.5 --num-processes 16  --num-steps 20 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --lrCL $lrCL --choiceCLparams 0 --seed $seed --expansion $expansion --pycolab_apple_reward_min 0. --pycolab_apple_reward_max 0. --pycolab_final_reward 1. --factorR $factorR
python -u mainCLdelaysep84TVTofficialDoina.py  --algo ppoCLsep84  --factorC $factorC --use-gae --lr $lr --clip-param 0.08 --value-loss-coef 0.5 --num-processes 16  --num-steps 20 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.02 --lrCL $lrCL --choiceCLparams 0 --seed $seed --expansion $expansion --pycolab_apple_reward_min 0. --pycolab_apple_reward_max 0. --pycolab_final_reward 1. --factorR $factorR  --head 8



#expansion=$1
#seed=$2
#lrCL=$3
#factorR=$4
#factorC=$5
#python3 -u mainCLdelaysep84TVTregress.py --algo ppoCLsep84  --use-gae --lr 2e-4 --clip-param 0.08 --value-loss-coef 0.5 --num-processes 16 --num-steps 20 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --lrCL 10e-4 --choiceCLparams 0 --seed  $seed  --head 10 --expansion $expansion  --pycolab_apple_reward_min 0. --pycolab_apple_reward_max 0. --pycolab_final_reward 1.

#python3 -u $algo --expansion $expansion --seed $seed --AlgoType 6 --head 8 --beta 1. --phrase 2 -cosB 100. --costB 10. --entropy_cost .1 --training 1 --moduletype 1 --batch_size 16 --batch_sizeSUCCESS 16 --with_memoryx 0 --do_tvt 0 --usepos 0 --pycolab_apple_reward_min 0. --pycolab_apple_reward_max 0. --pycolab_final_reward 10. --learning_rate 10e-4 --learning_rate2 20e-4
#python3 -u mainCLdelaysep84TVTregress.py --algo ppoCLsep84  --use-gae --lr 2e-4 --clip-param 0.08 --value-loss-coef 0.5 --num-processes 16 --num-steps 20 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --lrCL $lrCL --factorR $factorR  --factorC $factorC --choiceCLparams 0 --seed  $seed  --head 10 --expansion $expansion  --pycolab_apple_reward_min 0. --pycolab_apple_reward_max 0. --pycolab_final_reward 10.
