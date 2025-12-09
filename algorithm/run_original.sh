TRACE_META_DIR="./output"
OUTPUT_PATH="./logs_original_accuracy/"
LIMIT_ARG=""
INT8_MODE=false

if [ "$INT8_MODE" = true ]; then
    INT8_ARG="--load_in_8bit"
else
    INT8_ARG=""
fi

python -m run_eval --model llava_vid --model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average --tasks videomme --batch_size 1 --log_samples --log_samples_suffix llava_vid --output_path $OUTPUT_PATH $INT8_ARG --write_accuracy --trace_meta_dir $TRACE_META_DIR $LIMIT_ARG
python -m run_eval --model llava_vid --model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average --tasks mlvu --batch_size 1 --log_samples --log_samples_suffix llava_vid --output_path $OUTPUT_PATH $INT8_ARG --write_accuracy --trace_meta_dir $TRACE_META_DIR $LIMIT_ARG
python -m run_eval --model llava_vid --model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average --tasks mvbench --batch_size 1 --log_samples --log_samples_suffix llava_vid --output_path $OUTPUT_PATH $INT8_ARG --write_accuracy --trace_meta_dir $TRACE_META_DIR $LIMIT_ARG
python -m run_eval --model llava_onevision --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen --tasks videomme --batch_size 1 --log_samples --log_samples_suffix llava_onevision --output_path $OUTPUT_PATH $INT8_ARG --write_accuracy --trace_meta_dir $TRACE_META_DIR $LIMIT_ARG
python -m run_eval --model llava_onevision --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen --tasks mlvu --batch_size 1 --log_samples --log_samples_suffix llava_onevision --output_path $OUTPUT_PATH $INT8_ARG --write_accuracy --trace_meta_dir $TRACE_META_DIR $LIMIT_ARG
python -m run_eval --model llava_onevision --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen --tasks mvbench --batch_size 1 --log_samples --log_samples_suffix llava_onevision --output_path $OUTPUT_PATH $INT8_ARG --write_accuracy --trace_meta_dir $TRACE_META_DIR $LIMIT_ARG
python -m run_eval --model minicpm_v --model_args pretrained=openbmb/MiniCPM-V-2_6 --tasks videomme --batch_size 1 --log_samples --log_samples_suffix minicpm_v --output_path $OUTPUT_PATH $INT8_ARG --write_accuracy --trace_meta_dir $TRACE_META_DIR $LIMIT_ARG
python -m run_eval --model minicpm_v --model_args pretrained=openbmb/MiniCPM-V-2_6 --tasks mlvu --batch_size 1 --log_samples --log_samples_suffix minicpm_v --output_path $OUTPUT_PATH $INT8_ARG --write_accuracy --trace_meta_dir $TRACE_META_DIR $LIMIT_ARG
python -m run_eval --model minicpm_v --model_args pretrained=openbmb/MiniCPM-V-2_6 --tasks mvbench --batch_size 1 --log_samples --log_samples_suffix minicpm_v --output_path $OUTPUT_PATH $INT8_ARG --write_accuracy --trace_meta_dir $TRACE_META_DIR $LIMIT_ARG