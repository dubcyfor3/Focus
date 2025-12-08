TRACE_META_DIR="./output"

# Check arguments
FULL_MODE=false
INT8_MODE=false

for arg in "$@"; do
    if [ "$arg" = "full" ]; then
        FULL_MODE=true
    elif [ "$arg" = "int8" ]; then
        INT8_MODE=true
    fi
done

# Set trace directory based on int8 mode
if [ "$INT8_MODE" = true ]; then
    TRACE_DIR="${TRACE_META_DIR}/focus_int8/"
else
    TRACE_DIR="${TRACE_META_DIR}/focus_main/"
fi

# Set quantization argument
if [ "$INT8_MODE" = true ]; then
    INT8_ARG="--load_in_8bit"
else
    INT8_ARG=""
fi

# Check if first argument is "full"
if [ "$FULL_MODE" = true ]; then
    if [ "$INT8_MODE" = true ]; then
        OUTPUT_PATH="./logs_focus_accuracy_int8/"
    else
        OUTPUT_PATH="./logs_focus_accuracy/"
    fi
    # Full run: no limit, for accuracy measurement
    TRACE_ARGS=""
    LIMIT_ARG=""
    WRITE_ARGS="--write_accuracy"
else
    # Default: limit 10, for sparse trace generation
    # Build TRACE_ARGS conditionally: use_median only if int8 is NOT enabled
    if [ "$INT8_MODE" = true ]; then
        TRACE_ARGS="--export_focus_trace"
    else
        TRACE_ARGS="--export_focus_trace --use_median"
    fi
    LIMIT_ARG="--limit 10"
    if [ "$INT8_MODE" = true ]; then
        OUTPUT_PATH="./logs_focus_traces_int8/"
    else
        OUTPUT_PATH="./logs_focus_traces/"
    fi
    WRITE_ARGS=""
fi


# Original command:
# python -m run_eval --model llava_vid --model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average --tasks videomme --focus --batch_size 1 --log_samples --log_samples_suffix llava_vid --output_path ./logs_traces/ --limit 10 --export_focus_trace --trace_dir ./output/focus_main/ --trace_name llava_vid_videomme --use_median --trace_meta_dir ./output/

python -m run_eval --model llava_vid --model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average --tasks videomme --focus --batch_size 1 --log_samples --log_samples_suffix llava_vid --output_path $OUTPUT_PATH $LIMIT_ARG $TRACE_ARGS --trace_name llava_vid_videomme $INT8_ARG --trace_dir $TRACE_DIR --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model llava_vid --model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average --tasks mlvu --focus --batch_size 1 --log_samples --log_samples_suffix llava_vid --output_path $OUTPUT_PATH $LIMIT_ARG $TRACE_ARGS --trace_name llava_vid_mlvu $INT8_ARG --trace_dir $TRACE_DIR --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model llava_vid --model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average --tasks mvbench --focus --batch_size 1 --log_samples --log_samples_suffix llava_vid --output_path $OUTPUT_PATH $LIMIT_ARG $TRACE_ARGS --trace_name llava_vid_mvbench $INT8_ARG --trace_dir $TRACE_DIR --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model llava_onevision --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen --tasks videomme --focus --batch_size 1 --log_samples --log_samples_suffix llava_onevision --output_path $OUTPUT_PATH $LIMIT_ARG $TRACE_ARGS --trace_name llava_onevision_videomme $INT8_ARG --trace_dir $TRACE_DIR --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model llava_onevision --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen --tasks mlvu --focus --batch_size 1 --log_samples --log_samples_suffix llava_onevision --output_path $OUTPUT_PATH $LIMIT_ARG $TRACE_ARGS --trace_name llava_onevision_mlvu $INT8_ARG --trace_dir $TRACE_DIR --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model llava_onevision --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen --tasks mvbench --focus --batch_size 1 --log_samples --log_samples_suffix llava_onevision --output_path $OUTPUT_PATH $LIMIT_ARG $TRACE_ARGS --trace_name llava_onevision_mvbench $INT8_ARG --trace_dir $TRACE_DIR --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model minicpm_v --model_args pretrained=openbmb/MiniCPM-V-2_6 --tasks videomme --focus --batch_size 1 --log_samples --log_samples_suffix minicpm_v --output_path $OUTPUT_PATH $LIMIT_ARG $TRACE_ARGS --trace_name minicpm_v_videomme $INT8_ARG --trace_dir $TRACE_DIR --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model minicpm_v --model_args pretrained=openbmb/MiniCPM-V-2_6 --tasks mlvu --focus --batch_size 1 --log_samples --log_samples_suffix minicpm_v --output_path $OUTPUT_PATH $LIMIT_ARG $TRACE_ARGS --trace_name minicpm_v_mlvu $INT8_ARG --trace_dir $TRACE_DIR --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model minicpm_v --model_args pretrained=openbmb/MiniCPM-V-2_6 --tasks mvbench --focus --batch_size 1 --log_samples --log_samples_suffix minicpm_v --output_path $OUTPUT_PATH $LIMIT_ARG $TRACE_ARGS --trace_name minicpm_v_mvbench $INT8_ARG --trace_dir $TRACE_DIR --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS