for i in kkt_od_sft_fp16_fp16 kor_wiki_quad_od_instruct_f16 f16_instruction_tuning_synth_real_sft kkt_instruction_tuning_sft_synth_sft_simpo_real_f16 kkt_instruction_tune_synth_sft_synth_simpo_f16; do tsp python src/evaluate/evaluate.py --model_name $i; done
for i in kkt_cd_fp16_fp16 kkt_corpus_cd_sft_fp16; do tsp python src/evaluate/evaluate.py --model_name $i --dataset_name kkt_cd_inst; done
