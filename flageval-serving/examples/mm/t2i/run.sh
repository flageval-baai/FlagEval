#!/bin/bash

current_file="$0"
current_dir="$(dirname "$current_file")"
TASK_NAME=$1
SERVER_IP=$2
SERVER_PORT=$3
PYTHONPATH=$current_dir:$PYTHONPATH python model_adapter.py --task $TASK_NAME --server_ip $SERVER_IP --server_port $SERVER_PORT