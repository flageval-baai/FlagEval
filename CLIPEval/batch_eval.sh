#!/bin/bash

while read model
do
    echo $model
    python evaluate.py --model_name=$model --model_dir=/sharefs/bowen/ckpt_mclip --output=outputs/$model.json
done < $1