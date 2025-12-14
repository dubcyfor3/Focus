# ***Focus***: A Streaming Concentration Architecture for Efficient Vision-Language Models

***Focus*** is a hardware–algorithm co-designed architecture that accelerates Vision-Language Model (VLM) inference by eliminating redundancy in visual tokens. It introduces a **multilevel concentration** pipeline—covering semantic-, block-, and vector-level redundancy—to reorganize VLM inputs into a hardware-friendly, locality-preserving format. An overview of ***Focus*** design is shown below

<p align="center">
<iframe src="./focus_overview.png" width="800" height="600"></iframe>
</p>
<p align="center">
*Focus* Architecture Overview
</p>

This repository provides a full-stack implementation of *Focus*, including the algorithm, architecture simulator, RTL hardware design, and baselines. It reproduces all experimental results reported in our evaluation.

---

## **Overview**

> 📚 Component Documentation
>
> This repository contains three major components, each with a detailed README:
>
> * **[Algorithm](algorithm/README.md)** — *Focus* algorithm, sparse-trace generation, and accuracy evaluation.
> * **[Simulator](simulator/README.md)** — Performance modeling and design-space exploration.
> * **[RTL Hardware](rtl/README.md)** — Verilog implementation of Focus hardware modules.

---

## **Repository Structure**

* **`algorithm/`** – *Focus* algorithm implementation and accuracy evaluation.

* **`simulator/`** – Architecture performance simulator.

* **`rtl/`** – Hardware RTL implementation
  Includes systolic array, SEC/SIC, and other hardware blocks.

* **`evaluation_scripts/`** – Plotting and result-analysis utilities

  * `plot_scripts/` — Jupyter notebooks for generating paper figures.

* **`3rd_party/`** – Third-party dependencies

  * `LLaVA-NeXT/` – LLaVA VLM implementation
  * `scalesim/` – GEMM performance simulator
  * `cacti/` – SRAM memory modeling
  * `DRAMsim3/` – DRAM simulation

---

## **Getting Started**

### **Prerequisites**

* Python **3.11** (conda recommended)
* CUDA-capable GPU (**≥80 GB HBM recommended**)
* G++
* HuggingFace access token (for model checkpoints and datasets)

---

## **Installation**

1. **Clone the repository**

```bash
git clone git@github.com:dubcyfor3/Focus.git
cd Focus
```

2. **Initialize submodules**

```bash
git submodule init
git submodule update
```

3. **Create and activate the environment**

```bash
conda create -n focus python=3.11 -y
conda activate focus
```

4. **Install dependencies**

```bash
# Install LLaVA-NeXT
cd 3rd_party/LLaVA-NeXT
pip install -e .

# Install ScaleSim
cd ../scalesim
pip install -e .

# Build CACTI
cd ../cacti
make

# Build DRAMsim3
cd ../DRAMsim3
make

# Install lmms-eval
cd ../../algorithm/lmms-eval
pip install -e .

# Install *Focus*
cd ../focus
pip install -e '.[main]'   # '[main]' ensures the correct transformers version
# pip install -e '.[qwen25_vl]' # run this when running QWen2.5-VL
```

---

## **Running VLMs with *Focus***

### **1. Algorithm: Generate Sparse Traces & Evaluate Accuracy**

Example command to run LLaVA-Video with Focus on VideoMME dataset and export sparse traces:

```bash
cd algorithm/
python -m run_eval \
  --model llava_vid \
  --model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average \
  --tasks videomme \
  --focus \
  --batch_size 1 \
  --log_samples --log_samples_suffix llava_vid \
  --output_path ./logs_traces/ \
  --limit 10 \
  --export_focus_trace \
  --trace_dir ./output/focus_main/ \
  --trace_name llava_vid_videomme \
  --use_median \
  --trace_meta_dir ./output/
```

See *[algorithm/README.md](algorithm/README.md)* for complete usage and scripts for all datasets and models.

---

### **2. Simulator: Run Architecture Simulation**

Example simulation using generated traces:

```bash
cd ../simulator
python main.py \
  --model llava_vid \
  --dataset videomme \
  --accelerator focus \
  --trace_dir ../algorithm/output \
  --output_dir results
```

See *[simulator/README.md](simulator/README.md)* for details on all experiments and configurations.

---

### **3. Evaluation Scripts: Plot Figures & Tables**

```bash
cd ../evaluation_scripts/plot_scripts/ipynb_src
# Open the Jupyter notebooks and execute to generate plots
```

The notebooks provide end-to-end instructions for reproducing all figures and tables from the paper.

---

## **Acknowledgement**

This repository is built on top of the following open-source projects:

- **[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)**
- **[LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)**
- **[FrameFusion](https://github.com/thu-nics/FrameFusion)**
- **[ScaleSim](https://github.com/scalesim-project/SCALE-Sim)**
- **[CACTI](https://github.com/HewlettPackard/cacti)**
- **[DRAMsim3](https://github.com/umd-memsys/DRAMsim3)**

We thank the authors and contributors of these projects for their valuable work.

---

Thank you for using ***Focus***!
If you have questions or issues, please feel free to open an issue or pull request.
