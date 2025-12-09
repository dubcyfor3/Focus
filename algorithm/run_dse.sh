TRACE_META_DIR="./output/"

# Check arguments
ACCURACY_MODE=false

for arg in "$@"; do
    if [ "$arg" = "accuracy" ]; then
        ACCURACY_MODE=true
    fi
done

# Check if first argument is "accuracy"
if [ "$ACCURACY_MODE" = true ]; then
    # Accuracy mode: 500 samples for rough accuracy measurement
    TRACE_ARGS=""
    LIMIT_ARG="--limit 500"
    OUTPUT_PATH="./logs_dse_accuracy/"
    WRITE_ARGS="--write_accuracy"
else
    # Default: limit 10, for sparse trace generation
    TRACE_ARGS="--export_focus_trace"
    LIMIT_ARG="--limit 10"
    OUTPUT_PATH="./logs_dse_trace/"
    WRITE_ARGS=""
fi


TRACE_DIR="${TRACE_META_DIR}/m_tile_size_dse/"

for gemm_m_size in -1 4096 2048 1024 512 128 32; do
    python -m run_eval --model llava_vid --model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average --tasks videomme --focus --batch_size 1 --log_samples --log_samples_suffix llava_vid --output_path $OUTPUT_PATH $LIMIT_ARG $TRACE_ARGS --trace_dir $TRACE_DIR --trace_name llava_vid_videomme_${gemm_m_size} --trace_meta_dir $TRACE_META_DIR --gemm_m_size $gemm_m_size $WRITE_ARGS --write_accuracy_table_name dse_a_m_tile_accuracy.csv
done

TRACE_DIR="${TRACE_META_DIR}/block_size_dse/"

for temporal_block_size in 1 2 3; do
    for spatial_block_size in 1 2 3; do
        python -m run_eval --model llava_vid --model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average --tasks videomme --focus --batch_size 1 --log_samples --log_samples_suffix llava_vid --output_path $OUTPUT_PATH $LIMIT_ARG $TRACE_ARGS --trace_dir $TRACE_DIR --trace_name llava_vid_videomme_${temporal_block_size}x${spatial_block_size}x${spatial_block_size} --trace_meta_dir $TRACE_META_DIR --frame_block_size $temporal_block_size --block_size $spatial_block_size $WRITE_ARGS --write_accuracy_table_name dse_c_block_accuracy.csv
    done
done

TRACE_DIR="${TRACE_META_DIR}/vector_size_dse/"

for vector_size in 4096 2048 512 128 32 8; do
    python -m run_eval --model llava_vid --model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average --tasks mlvu --focus --batch_size 1 --log_samples --log_samples_suffix llava_vid --output_path $OUTPUT_PATH $LIMIT_ARG $TRACE_ARGS --trace_dir $TRACE_DIR --trace_name llava_vid_mlvu_${vector_size} --trace_meta_dir $TRACE_META_DIR --vector_size $vector_size $WRITE_ARGS --write_accuracy_table_name dse_b_vector_accuracy.csv
done