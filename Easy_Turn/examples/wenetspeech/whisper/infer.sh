dir=./examples/wenetspeech/whisper/exp/interrupt  #存放模型的本地路径，需要先进行模型合并 
gpu_id=6 #单卡推理
test_data_dir='data' #测试集的大路径
test_sets='interrupt_test' #测试集的小路径
ckpt_name=epoch_0.pt #checkpoint的名称
task='<TRANSCRIBE><BACKCHANNEL><COMPLETE>' # task名称，详见conf/prompt.yaml
data_type='shard_full_data' # raw  shard_full_data 两种类型可选，与训练相同

bash decode/decode_common.sh \
    --data_type $data_type \
    --test_sets "$test_sets" \
    --test_data_dir $test_data_dir \
    --gpu_id $gpu_id \
    --dir $dir \
    --ckpt_name $ckpt_name \
    --task "$task" 
