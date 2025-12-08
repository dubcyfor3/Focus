TRACE_META_DIR="./output/"
if [ "$1" = "full" ]; then
    LIMIT_ARG=""
    WRITE_ARGS="--write_accuracy"
else
    LIMIT_ARG="--limit 10"
    WRITE_ARGS="--write_sparsity"
fi

python -m run_eval --model llava_vid --model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average --tasks videomme --CMC --batch_size 1 --log_samples --log_samples_suffix llava_vid --output_path ./logs_traces/ $LIMIT_ARG --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model llava_vid --model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average --tasks mlvu --CMC --batch_size 1 --log_samples --log_samples_suffix llava_vid --output_path ./logs_traces/ $LIMIT_ARG --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model llava_vid --model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average --tasks mvbench --CMC --batch_size 1 --log_samples --log_samples_suffix llava_vid --output_path ./logs_traces/ $LIMIT_ARG --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model llava_onevision --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen --tasks videomme --CMC --batch_size 1 --log_samples --log_samples_suffix llava_onevision --output_path ./logs_traces/ $LIMIT_ARG --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model llava_onevision --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen --tasks mlvu --CMC --batch_size 1 --log_samples --log_samples_suffix llava_onevision --output_path ./logs_traces/ $LIMIT_ARG --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model llava_onevision --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen --tasks mvbench --CMC --batch_size 1 --log_samples --log_samples_suffix llava_onevision --output_path ./logs_traces/ $LIMIT_ARG --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model minicpm_v --model_args pretrained=openbmb/MiniCPM-V-2_6 --tasks videomme --CMC --batch_size 1 --log_samples --log_samples_suffix minicpm_v --output_path ./logs_traces/ $LIMIT_ARG --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model minicpm_v --model_args pretrained=openbmb/MiniCPM-V-2_6 --tasks mlvu --CMC --batch_size 1 --log_samples --log_samples_suffix minicpm_v --output_path ./logs_traces/ $LIMIT_ARG --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model minicpm_v --model_args pretrained=openbmb/MiniCPM-V-2_6 --tasks mvbench --CMC --batch_size 1 --log_samples --log_samples_suffix minicpm_v --output_path ./logs_traces/ $LIMIT_ARG --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS