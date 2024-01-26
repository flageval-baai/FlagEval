#!/bin/bash
work_dir=`dirname $0`
work_dir=`cd $work_dir;pwd`
cd $work_dir

# (重要变量)输出结果的路径。只有将结果写入这个路径，平台才能获取到结果。使用方法示例:
# python main.py > $FLAGEVAL_OUTPUT_PATH
# python main.py > ./res.json; cp res.json $FLAGEVAL_OUTPUT_PATH
echo $FLAGEVAL_OUTPUT_PATH

# 如果训练数据过多，可以通过flageval-serving上传训练数据的压缩包，然后在此run.sh里面解压
# tar -zxvf mydata.tar.gz &>/dev/null
# 写你的训练逻辑
# torchrun --nnodes=$FLAGEVAL_NODE_NUM test.py ... > res.json
echo "running..."

# 此处只是demo，"./res.json"应该替换为实际结果
cp ./res.json $FLAGEVAL_OUTPUT_PATH

####################################################################################
# 如果是分布式集群训练，那么以下变量可根据情况使用

# master节点的数量
echo $FLAGEVAL_MASTER_COUNT

# worker节点的数量
echo $FLAGEVAL_WORKER_COUNT

# master节点的ip地址。如果有多个master节点，则变量为第一个master的ip
echo $FLAGEVAL_MASTER_IP

# 当前节点是第几个节点。所有master节点的值都为0。worker节点的num为1, 2, 3...
echo $FLAGEVAL_NODE_NUM

# 当前节点是master还是worker. 值只可能为 "master" 或 "worker"
echo $FLAGEVAL_NODE
# 示例:
if [ "x$FLAGEVAL_NODE" == "xmaster" ]; then
    echo "I am master"
else
    echo "I am worker"
fi

