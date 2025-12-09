# Algorithm Implementation

This directory contains the algorithm implementation for Focus, including trace generation, accuracy evaluation, and baseline comparisons.

These components realize the following things:
- **Focus algorithm**: Core multilevel concentration algorithm. It compresses the visual tokens prior to each GEMM operation (FC layer or attention) in VLM. This is realized by creating a replica of the original model forward function and inserting Focus concentration algorithm within the forward function.
- **Baseline algorithm**: We implement the algorithm of CMC and Adaptiv (our baselines) and use the same way as Focus to apply it in VLM. For Frame Fusion baseline, we directly use their official implementation.
- **Accuracy evaluation**: We integrate these algorithms into lmms-eval evaluation framework to measure how VLM capabilities change as different algorithms are applied.
- **Trace generation**: We record the sparse trace induced by Focus and baselines for later simulation. We directly record the overall sparsity induced by CMC and Adaptiv since they both generate coarse-grained token level sparsity. For Focus, we record the fine-grained vector-wise sparse trace and save it for simulation.

## Directory Structure

- `focus/` - Focus and baselines implementation
    - `main.py` - Focus algorithm implementation in class `Focus`
    - `baseline_CMC.py` - CMC implementation
    - `baseline_adaptiv.py` - Adaptiv implementation
    - `interface.py` - utilities used to replace original model forward function with forward integrated with Focus
    - `models/` - new model forward definition considering Focus or baselines
- `lmms-eval/` - VLM evaluation framework, measure accuracy
- `example_output/` - example of algorithm evaluation output, contains VLM input metadata, accuracy and sparsity (does not contain Focus sparse traces)
- `run_eval.py` - main program to launch evaluation, run with argument `--focus` or `--frame_fusion` or `--adaptiv` or `--CMC` for inference with corresponding method
- `run_*.sh` - execution scripts for comprehensive trace generation and accuracy evaluation

## Supported Models

Focus currently supports several typical VLMs and can be easily extended to new VLMs following the modification example in `models/qwen2/modeling_qwen2.py`.

Currently supported models on HuggingFace Transformers (users may need to request access on HuggingFace to access these models):
- [LLaVA-Video](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2) (`lmms-lab/LLaVA-Video-7B-Qwen2`)
- [Llava-OneVision](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov) (`lmms-lab/llava-onevision-qwen2-7b-ov`)
- [MiniCPM-V-2.6](https://huggingface.co/openbmb/MiniCPM-V-2_6) (`openbmb/MiniCPM-V-2_6`)
- [QWen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) (`Qwen/Qwen2.5-VL-7B-Instruct`)

## Reproducing Results
We provide shell scripts that call the Python program `run_eval.py` with specific argument to run various models and datasets with Focus and baseline algorithms. 
Users may specify `TRACE_META_DIR` in shell scripts to decide which path to store the trace and accuracy.

### Trace Generation (Estimated time: 6 GPU hours)

#### Focus Trace Generation

Generate sparse traces for Focus on Llava-Video, Llava-OneVision and MiniCPMV-2.6 on VideoMME, MLVU, and MVBench datasets. We take the first 10 samples (`--limit 10`) and save the trace of the median sample.
```bash
sh run_focus.sh     # (Estimated time: 1 hour, may require longer time to download models and datasets for the first time)
```

#### Baseline Trace Generation

Measure sparsity of baselines.
```bash
sh run_adaptiv.sh # (Estimated time: 25 minutes)
sh run_cmc.sh     # (Estimated time: 2 hours and 40 minutes)
```

> Note: the execution time on GPU does not reflect the efficiency of these methods on specialized hardware.

#### Focus Design Space Exploration Trace Generation

We conduct comprehensive design space exploration for Focus. We scan over the hyperparameters of Focus that influence the sparse trace of Focus.
- GEMM m tile size
- Vector size
- Block size

```bash
sh run_dse.sh   # (Estimated time: 70 minutes)
```


#### INT8 Focus Trace Generation

Quantize the model to INT8 and evaluate the influence on Focus.
```bash
sh run_focus.sh int8  # (Estimated time: 70 minutes)
```


#### Focus on image-input VLMs Trace Generation

Although oriented for VLM with video input, Focus can also be applied to VLM with single-image input. Run Focus and baseline on Llava-OneVision and QWen2.5-VL on VQAv2, MME, and MMBench.
Configure the environment before running this part to support QWen2.5-VL:
```
cd focus/
pip install -e '.[qwen25_vl]'
```

```bash
sh run_focus_image.sh   # (Estimated time: 5 minutes, may require longer time to download models and datasets for the first time)
sh run_adaptiv_image.sh # (Estimated time: 3 minutes)
```

> Note: run `pip install -e '.[main]'` again after finishing image-input VLM evaluation

#### Trace Output Format
After the aforementioned trace generation process, traces are saved as pth files and CSV files under specified directory, containing:
- `focus_main`
- `m_tile_size_dse`
- `vector_size_dse`
- `block_size_dse`
- `focus_int8`
- `meta_data.csv`
- `adaptiv_sparsity.csv`
- `cmc_sparsity.csv`

---

### Accuracy Evaluation (Optional) (Estimated time: 480 GPU hours)

Accuracy evaluation uses the full dataset (no limit) and measures model performance on downstream tasks. Since it is evaluated on full dataset, it takes a long time to run and get accuracy results. Therefore, we provide expected results of accuracy evaluation under `example_output`. Skipping accuracy evaluation will not influence the process of architecture simulation.

#### Main Accuracy Evaluation

Measure model accuracy of various methods.
- Original model accuracy
- Inference with Focus
- Inference with FrameFusion
- Inference with Adaptiv
- Inference with CMC

```bash
sh run_original.sh full     # (Estimated time: 31 hours)
sh run_focus.sh full     # (Estimated time: 66 hours)
sh run_framefusion.sh full   # (Estimated time: 31 hours)
sh run_adaptiv.sh full   # (Estimated time: 33 hours)
sh run_cmc.sh full       # (Estimated time: 242 hours)
```

Results saved to `accuracy.csv` under specified directory

#### Accuracy in DSE

We also roughly measure how Focus hyperparameters influence accuracy by taking 500 samples. Results saved to `dse_*_accuracy.csv`
```bash
sh run_dse.sh accuracy  # (Estimated time: 11 hours)
```

#### INT8 Quantization Accuracy

Evaluate Focus with INT8 quantization for reduced precision inference. Results will be saved to `accuracy.csv`
```bash
sh run_focus.sh int8 full   # (Estimated time: 70 hours)
sh run_original.sh int8 full  # (Estimated time: 35 hours)
```

#### VLM Accuracy on Single-Image Tasks
We take the first 1000 samples for each dataset for accuracy evaluation of single-image tasks. Results will be saved to `accuracy.csv`
Please also configure environment using `pip install -e '.[qwen25_vl]'`


```bash
sh run_original_image.sh accuracy   # (Estimated time: 70 minutes)
sh run_focus_image.sh accuracy      # (Estimated time: 6 hours)
sh run_adaptiv_image.sh accuracy    # (Estimated time: 70 minutes)
```

---
The output of this part will be used by `simulator/` or `evaluation_scripts/` to reproduce results in paper.