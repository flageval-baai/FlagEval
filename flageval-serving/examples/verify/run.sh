#!/bin/bash
work_dir=`dirname $0`
work_dir=`cd $work_dir;pwd`
cd $work_dir

# master节点的数量
echo $FLAGEVAL_MASTER_COUNT

# worker节点的数量
echo $FLAGEVAL_WORKER_COUNT

# master节点的ip地址。如果有多个master节点，则变量为第一个master的ip
echo $FLAGEVAL_MASTER_IP

# 当前节点是master还是worker. 值只可能为 "master" 或 "worker"
echo FLAGEVAL_NODE

# 当前节点是第几个节点。所有master节点的值都为0。worker节点的num为1, 2, 3...
echo $FLAGEVAL_NODE_NUM

# 输出结果的路径。如果get_res.py为输出结果json的文件，则可以写为:python get_res.py > $FLAGEVAL_OUTPUT_PATH
echo $FLAGEVAL_OUTPUT_PATH

# 写你的训练逻辑
# torchrun --nnodes=$FLAGEVAL_NODE_NUM test.py ...
echo "running..."

# 如果是master节点，则上传结果到$FLAGEVAL_OUTPUT_PATH。
if [ "x$FLAGEVAL_NODE" == "xmaster" ]; then
    # 此处只是demo，"targets/detection.json"应该替换为实际结果
    cp targets/detection.json $FLAGEVAL_OUTPUT_PATH
fi
