#!/bin/bash
# 评估集路径
# --eval_set /cosmos/local/users/zifanwang/SpamLLM/data/${input_file_name} \
# --eval_set /cosmos/local/IndexQuality/FinetuneLLM/EvaluationSets/${input_file_name} \
eval_set_dir="/cosmos/local/IndexQuality/FinetuneLLM/EvaluationSets/"
# eval_set_dir="/cosmos/local/users/zifanwang/SpamLLM/data/"

# 定义输入文件列表
input_file_names=("scrapekr1.2_spamllm2.4.parquet" "test_dataset_2024_03_05.tsv" "spamgtx5.0_UHRSoutput.parquet" "Clean_60k.tsv" "8k_with_flipped_labels.tsv")
# input_file_names=("scrapekr1.2_spamllm2.4.parquet")
# input_file_names=("scrapekr1.2_spamllm2.4_UrlExpectedLabel_escape.tsv" "test_dataset_2024_03_05_escape.tsv" "spamgtx5.0_UHRSoutput_escape.tsv" "Clean_60k_escape.tsv" "8k_with_flipped_labels_escape.tsv" "auditor_ym_escape1.tsv") # from zifan
# input_file_names=("auditor_ym_escape1.tsv")
# input_file_names=("scrapekr1.2_spamllm2.4_UrlExpectedLabel_escape.tsv")

# 定义输入模型列表
# Qwen
source_model_dir="/cosmos/local/IndexQuality/FinetuneLLM/FullTrainTest/qwen_no_overlap_o1_a3_v1/"

# Find the best model
# for i in {200..29200..200};do
#     echo "Current model is: model_${i}"
#     for input_file_name in "${input_file_names[@]}"; do
#         # 记录当前文件名
#         echo "Current processing file is: ${input_file_name}..."
#         # 执行命令
#         export CUDA_VISIBLE_DEVICES="1,2,3"
#         NCCL_DEBUG=WARN python -m torch.distributed.run  \
#         --nnodes 1 --nproc_per_node 3 inf_qwen_v1.py \
#         --model_path "$source_model_dir/model_${i}/pytorch_model.bin" \
#         --eval_set "${eval_set_dir}/${input_file_name}" \
#         --max_seq_length 1024 \
#         --output_dir "$source_model_dir/model_${i}/${input_file_name}" \
#         --batch_size 32 
#     done
# done


# Evaluation on 5 eval-sets
i=4200
echo "Current model is: current_best_${i}"
for input_file_name in "${input_file_names[@]}"; do
    # 记录当前文件名
    echo "Current processing file is: ${input_file_name}..."

    export CUDA_VISIBLE_DEVICES="1,2,3"
    NCCL_DEBUG=WARN python -m torch.distributed.run  \
    --nnodes 1 --nproc_per_node 3 inf_qwen_v1.py \
    --model_path "$source_model_dir/model_${i}/pytorch_model.bin" \
    --eval_set "${eval_set_dir}/${input_file_name}" \
    --max_seq_length 1024 \
    --output_dir "$source_model_dir/model_${i}/${input_file_name}" \
    --batch_size 32 
done

# Evaluation several candidates on 1 Auditor
# candidates=("4200" "6600" "9000")
# for model_id in "${candidates[@]}"; do
#     for input_file_name in "${input_file_names[@]}"; do
#         # 记录当前文件名
#         echo "Current processing file is: ${input_file_name}..."
#         # 执行命令
#         export CUDA_VISIBLE_DEVICES="1,2,3"
#         NCCL_DEBUG=WARN python -m torch.distributed.run  \
#         --nnodes 1 --nproc_per_node 3 inf_qwen_v1.py \
#         --model_path "$source_model_dir/model_${model_id}/pytorch_model.bin" \
#         --eval_set "${eval_set_dir}/${input_file_name}" \
#         --max_seq_length 1024 \
#         --output_dir "$source_model_dir/model_${model_id}/${input_file_name}" \
#         --batch_size 32 
#     done
# done