DIR=${1}
MODEL_NAME=${2}
TRAIN=${3}
LR=${4}
torchrun --nproc_per_node 1 \
    -m FlagEmbedding.reranker.run \
    --output_dir $DIR \
    --model_name_or_path $MODEL_NAME \
    --train_data $TRAIN \
    --learning_rate $LR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --dataloader_drop_last True \
    --train_group_size 9 \
    --max_len 512 \
    --weight_decay 0.01 \
    --logging_steps 10 

mv $DIR /home/euiyul/