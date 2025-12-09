TRACE_DIR="../algorithm/output/"
OUTPUT_DIR="results"

python main.py --image_models_datasets --accelerator dense --trace_dir $TRACE_DIR --output_dir $OUTPUT_DIR
python main.py --image_models_datasets --accelerator adaptiv --trace_dir $TRACE_DIR --output_dir $OUTPUT_DIR
python main.py --image_models_datasets --accelerator focus --trace_dir $TRACE_DIR --output_dir $OUTPUT_DIR
