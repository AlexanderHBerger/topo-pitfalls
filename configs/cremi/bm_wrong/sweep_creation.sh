#!/bin/sh

for i in 1 2 3 4 5
do
    echo "$i"
    wandb sweep -p topo_pitfalls configs/cremi/bm_wrong/fold${i}.yaml
done
