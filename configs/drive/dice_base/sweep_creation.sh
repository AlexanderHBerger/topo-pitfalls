#!/bin/sh

for i in 1 2 3 4 5
do
    echo "$i"
    wandb sweep -p topo_pitfalls configs/drive/dice_base/fold${i}.yaml
done
