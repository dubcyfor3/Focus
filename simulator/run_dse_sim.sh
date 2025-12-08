TRACE_DIR="../algorithm/example_output/"
OUTPUT_DIR="test_sim_results"

python main.py --m_tile_size_dse --model llava_vid --dataset videomme --trace_dir $TRACE_DIR --output_dir $OUTPUT_DIR
python main.py --block_size_dse --model llava_vid --dataset videomme --trace_dir $TRACE_DIR --output_dir $OUTPUT_DIR
python main.py --vector_size_dse --model llava_vid --dataset mlvu --trace_dir $TRACE_DIR --output_dir $OUTPUT_DIR
python main.py --num_scatter_dse --model llava_vid --dataset videomme --trace_dir $TRACE_DIR --output_dir $OUTPUT_DIR