TRACE_META_DIR="./output"
TRACE_DIR="${TRACE_META_DIR}/focus_main/"
ACCURACY_MODE=False

for arg in "$@"; do
    if [ "$arg" = "accuracy" ]; then
        ACCURACY_MODE=true
    fi
done


if [ "$ACCURACY_MODE" = true ]; then
    LIMIT_ARG="--limit 1000"
    TRACE_ARGS=""
    OUTPUT_PATH="./logs_focus_image_accuracy/"
    WRITE_ARGS="--write_accuracy"
    echo "Evaluating accuracy on image tasks"
else
    LIMIT_ARG="--limit 10"
    TRACE_ARGS="--export_focus_trace --use_median"
    OUTPUT_PATH="./logs_focus_image_traces/"
    WRITE_ARGS=""
    echo "Exporting traces on image tasks"
fi

python -m run_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct --tasks vqav2 --focus --batch_size 1 --log_samples --log_samples_suffix llava_vid --output_path $OUTPUT_PATH $LIMIT_ARG $TRACE_ARGS --trace_dir $TRACE_DIR --trace_name qwen2_5_vl_vqav2 --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct --tasks mme --focus --batch_size 1 --log_samples --log_samples_suffix llava_vid --output_path $OUTPUT_PATH $LIMIT_ARG $TRACE_ARGS --trace_dir $TRACE_DIR --trace_name qwen2_5_vl_mme --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct --tasks mmbench --focus --batch_size 1 --log_samples --log_samples_suffix llava_vid --output_path $OUTPUT_PATH $LIMIT_ARG $TRACE_ARGS --trace_dir $TRACE_DIR --trace_name qwen2_5_vl_mmbench --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model llava_onevision --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen --tasks vqav2 --focus --batch_size 1 --log_samples --log_samples_suffix llava_onevision --output_path $OUTPUT_PATH $LIMIT_ARG $TRACE_ARGS --trace_dir $TRACE_DIR --trace_name llava_onevision_vqav2 --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model llava_onevision --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen --tasks mme --focus --batch_size 1 --log_samples --log_samples_suffix llava_onevision --output_path $OUTPUT_PATH $LIMIT_ARG $TRACE_ARGS --trace_dir $TRACE_DIR --trace_name llava_onevision_mme --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model llava_onevision --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen --tasks mmbench --focus --batch_size 1 --log_samples --log_samples_suffix llava_onevision --output_path $OUTPUT_PATH $LIMIT_ARG $TRACE_ARGS --trace_dir $TRACE_DIR --trace_name llava_onevision_mmbench --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS