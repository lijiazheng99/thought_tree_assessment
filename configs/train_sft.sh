# LLaMA:
CUDA_VISIBLE_DEVICES=0 python src/train.py \
    --stage sft \
    --do_train \
    --do_eval \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --dataset your_dataset_name_here \
    --val_size 2771 \
    --template llama3 \
    --finetuning_type lora \
    --lora_target all \
    --output_dir your_path_here \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 20 \
    --save_steps 100 \
    --eval_steps 100 \
    --do_sample False \
    --evaluation_strategy steps \
    --learning_rate 5e-5  \
    --num_train_epochs 4 \
    --plot_loss \
    --fp16 \
    --upcast_layernorm

# Mistral:
# CUDA_VISIBLE_DEVICES=$1 python src/train.py \
#     --stage sft \
#     --do_train \
#     --do_eval \
#     --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
#     --dataset your_dataset_name_here \
#     --val_size 2771 \
#     --template mistral \
#     --finetuning_type lora \
#     --lora_target all \
#     --output_dir your_path_here \
#     --overwrite_cache \
#     --per_device_train_batch_size 8 \
#     --gradient_accumulation_steps 8 \
#     --lr_scheduler_type cosine \
#     --logging_steps 20 \
#     --save_steps 100 \
#     --eval_steps 100 \
#     --do_sample False \
#     --evaluation_strategy steps \
#     --learning_rate 5e-5  \
#     --num_train_epochs 4 \
#     --plot_loss \
#     --quantization_bit 4 \
#     --fp16 \
#     --upcast_layernorm