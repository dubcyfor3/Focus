TRACE_META_DIR="./output/"
ACCURACY_MODE=False

for arg in "$@"; do
    if [ "$arg" = "accuracy" ]; then
        ACCURACY_MODE=true
    fi
done

if [ "$ACCURACY_MODE" = true ]; then
    LIMIT_ARG="--limit 1000"
    OUTPUT_PATH="./logs_adaptiv_image_accuracy/"
    WRITE_ARGS="--write_accuracy"
    echo "Evaluating accuracy on image tasks"
else
    LIMIT_ARG="--limit 10"
    WRITE_ARGS="--write_sparsity"
    OUTPUT_PATH="./logs_adaptiv_image_traces/"
    echo "Exporting traces on image tasks"
fi
python -m run_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct --tasks vqav2 --adaptiv --batch_size 1 --log_samples --log_samples_suffix llava_vid --output_path $OUTPUT_PATH $LIMIT_ARG --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct --tasks mme --adaptiv --batch_size 1 --log_samples --log_samples_suffix llava_vid --output_path $OUTPUT_PATH $LIMIT_ARG --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct --tasks mmbench --adaptiv --batch_size 1 --log_samples --log_samples_suffix llava_vid --output_path $OUTPUT_PATH $LIMIT_ARG --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model llava_onevision --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen --tasks vqav2 --adaptiv --batch_size 1 --log_samples --log_samples_suffix llava_onevision --output_path $OUTPUT_PATH $LIMIT_ARG --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model llava_onevision --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen --tasks mme --adaptiv --batch_size 1 --log_samples --log_samples_suffix llava_onevision --output_path $OUTPUT_PATH $LIMIT_ARG --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS
python -m run_eval --model llava_onevision --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen --tasks mmbench --adaptiv --batch_size 1 --log_samples --log_samples_suffix llava_onevision --output_path $OUTPUT_PATH $LIMIT_ARG --trace_meta_dir $TRACE_META_DIR $WRITE_ARGS