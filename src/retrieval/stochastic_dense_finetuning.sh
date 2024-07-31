torchrun --nproc_per_node 1\
    src/retrieval/stochastic_rag/FlagEmbedding/baai_general_embedding/finetune/run.py \
    --output_dir /mnt/c/Users/thddm/Documents/model/kkt-bge-m3-stochastic-dense \
    --model_name_or_path BAAI/bge-m3 \
    --train_data /mnt/c/Users/thddm/Documents/dataset/retrieval \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --dataloader_drop_last True \
    --normlized True \
    --temperature 0.02 \
    --query_max_len 54 \
    --passage_max_len 560 \
    --train_group_size 9 \
    --negatives_cross_device \
    --logging_steps 10 \
    --save_steps 1000 \
    --query_instruction_for_retrieval "" 