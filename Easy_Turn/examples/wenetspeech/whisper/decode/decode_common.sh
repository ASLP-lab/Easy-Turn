# 通用推理脚本
# cd ..
. ./path.sh || exit 1;

data_type= # raw  shard_full_data
test_sets=
test_data_dir=
gpu_id=
dir=
ckpt_name=  # 需要带pt
task= 
english_sets=''

lang=zh  # en zh
train_config=conf/train.yaml
average_checkpoint=false
average_num=5
average_mode=step
decode_modes="llmasr_decode"
decoding_chunk_size=-1
ctc_weight=0
reverse_weight=0.0
blank_penalty=0.0
length_penalty=0.0
batch_size=12
prompt_file=conf/prompt.yaml

. tools/parse_options.sh || exit 1;
# 将所有传进来的元素打印出来
echo "传入的参数为:"
echo 'data_type is '$data_type
echo 'test_sets is '$test_sets
echo 'test_data_dir is '$test_data_dir
echo 'gpu_id is '$gpu_id
echo 'dir is '$dir
echo 'ckpt_name is '$ckpt_name
echo 'task is '$task
echo 'print over'
echo 'prompt_file'$prompt_file



test_sets="${test_sets//---/ }"
echo "待推理的数据集为:"$test_sets
decode_checkpoint=$dir/$ckpt_name
export CUDA_VISIBLE_DEVICES="$gpu_id"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"

if [ ${average_checkpoint} == true ]; then
  decode_checkpoint=$dir/avg${average_num}_mode${average_mode}_max${max_step}.pt
  echo "do model average and final checkpoint is $decode_checkpoint"
  python wenet/bin/average_model.py \
    --dst_model $decode_checkpoint \
    --src_path $dir  \
    --num ${average_num} \
    --mode ${average_mode} \
    --max_step ${max_step} \
    --val_best
fi

for testset in ${test_sets}; do
  # shellcheck disable=SC2193
  if [[ "${data_type}" == "raw" ]]; then
    echo "data_type 为 raw"
    test_data="$test_data_dir/$testset/data.list"
  else
    echo "data_type 为 shard_full_data"
    test_data="$test_data_dir/$testset/shards_list.txt"
  fi
  echo "test_data is $test_data"

  if [[ " ${english_sets} " =~ " ${testset} " ]]; then
    lang="en"
  else
    lang="zh"
  fi
  echo "lang is $lang"
  base=$(basename $decode_checkpoint)
  result_dir=$dir/${testset}_${base}_chunk${decoding_chunk_size}_ctc${ctc_weight}_reverse${reverse_weight}_blankpenalty${blank_penalty}_lengthpenalty${length_penalty}
  python wenet/bin/recognize4llmasr.py --gpu ${gpu_id} \
    --modes $decode_modes \
    --config $dir/train.yaml \
    --data_type $data_type \
    --test_data $test_data \
    --checkpoint $decode_checkpoint \
    --beam_size 10 \
    --batch_size $batch_size \
    --blank_penalty ${blank_penalty} \
    --length_penalty ${length_penalty} \
    --ctc_weight $ctc_weight \
    --reverse_weight $reverse_weight \
    --result_dir $result_dir \
    --task $task \
    --lang $lang \
    ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}

  mkdir -p ${result_dir}
  for mode in ${decode_modes}; do
    python tools/compute-wer.py --char=1 --v=1 \
      $test_data_dir/$testset/text $result_dir/$mode/text > $result_dir/$mode/wer
  done
done

