#!/bin/sh

for i in 3 4 5
do
    echo "$i"
    wandb sweep -p topo_pitfalls configs/cremi/topograph/fold${i}.yaml
done
