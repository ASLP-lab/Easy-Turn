#!/bin/bash

# 数据路径
data_path="./raw.list" #json文件，格式参考训练第一步数据准备

# 任务种类
task="xxxx" #可设置每个tar包命名的前缀 

# 具体的tar文件存放位置以及索引文件位置
out_dir=./examples/wenetspeech/whisper/data
out_shards_txt_dir=./data/shards_list.txt

# 开始压缩，打包代码位于同级目录下
python ./make_shard_list.py \
    --num_utts_per_shard 1000 \
    --num_threads 8 \
    --resample 16000 \
    --prefix $task \
    $data_path \
    $out_dir \
    $out_shards_txt_dir

echo "Task: $task is done!"

# 参数解释：
#--num_utts_per_shard：每个 shard 中包含的 utterance 数量。
#--num_threads：处理 shard 的线程数量。
#--prefix：生成的 tar 文件名前缀。
#--resample：重采样率。
#$data_path：要进行压缩的data.list文件。
#$out_dir：压缩包输出路径。
#$output_shards_dir/$task/shards_list.txt：输出的包含所有压缩包路径的shards_list.txt文件路径。
