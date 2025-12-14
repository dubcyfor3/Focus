# Hardware Simulator

This directory contains the cycle-accurate simulator for evaluating *Focus* and baseline accelerators. The simulator receives sparse traces generated from algorithm components to derive accurate performance (execution time, energy) of accelerators.

This simulator provides cycle-accurate performance, energy, and area estimation for:
- ***Focus* accelerator**: Full *Focus* architecture with SEC and SIC units. The simulator supports *Focus* with only SEC enabled for ablation study.
- **Baseline accelerators**: Dense systolic array, Adaptiv, and CMC accelerators
- **Design space exploration**: Various hardware configurations including GEMM m tile size, vector size, block size, and number of scatter accumulators


## Directory Structure

- `main.py` - Entry point for running simulations
- `core/` - Core simulator implementation
- `arch/` - Accelerator architecture definitions
- `models/` - Model and workload definitions
- `memory/` - Memory modeling
- `utils/` - Utility modules
- `run_*.sh` - Execution scripts for different simulation scenarios

## Supported Architecture

- ***Focus***
- Vanilla Systolic Array
- CMC (ASPLOS 2024)
- AdapTiV (MICRO 2024)

## Simulator Basic Usage

Prior to running simulation, sparse traces should be generated. Basic arguments of the simulator:
 - `--accelerator` - architecture to simulate, select from `focus`, `adaptiv`, `cmc`, and `dense`
 - `--trace_dir` - path to the directory that stores all of the traces (no need to change path according to accelerator)
 - `--output_dir` - path to save simulation results
 - `--model` and `--dataset` - select traces for specific model and dataset

Example of running *Focus* simulation on Llava-Video, VideoMME:
```bash
python main.py --accelerator focus --trace_dir ../algorithm/output --model llava_vid --dataset videomme --output_dir results
```
The simulation will produce results including execution time, number of operations, energy breakdown, memory access, etc.


## Reproducing Results (Estimated time: 20 minutes)
We provide scripts to run simulation of *Focus* and baselines. Please set the `TRACE_DIR` to the `TRACE_META_DIR` used in trace generation in algorithm part, and set `OUTPUT_DIR` to your desired output directory. Simulation results may go through further processing in `evaluation_scripts` to get figures and tables in paper.

> Note: Our simulator invocate the scalesim during simulation, running multiple simulation at the same time may encounter bugs. We are working on solving this problem

### Main Simulation for *Focus* and Baselines

Run simulation of *Focus* and baselines on models and datasets we used:
```bash
sh run_main_sim.sh     # (Estimated time: 3 minutes)
```

Results saved to `main_focus.csv`, `main_dense.csv`, `main_adaptiv.csv`, and `main_cmc.csv`

### Architecture Specification Comparison (Table 3 in paper)

Compute the on-chip area and power of *Focus* and baselines using the statistics derived from TSMC memory compiler and synthesis result from Synopsys DC:
```bash
python arch/accelerator.py --output_dir OUTPUT_DIR    # (Estimated time: 5 seconds)
```

Results saved to `OUTPUT_DIR/accelerator_area_power_buffer.csv`

### Design Space Exploration Simulation

We conduct comprehensive design space exploration for *Focus*. We scan over different key hyper-parameters of *Focus* to evaluate its performance (we use cacti for adjustable buffer size evaluation):
- GEMM m tile size
- Vector size
- Block size
- Number of scatter accumulators

```bash
sh run_dse_sim.sh   # (Estimated time: 10 minutes)
```

Results saved to `dse_*.csv`

### *Focus* INT8 Simulation

Measure the sparsity induced by *Focus* using INT8 sparse traces:
```bash
python main.py --all_models_datasets --accelerator focus --quantization --trace_dir TRACE_DIR --output_dir OUTPUT_DIR     # (Estimated time: 3 minutes)
```

Results saved to `int8_focus.csv`

### Image-Input Models Simulation

Run simulation for image-input models:
```bash
sh run_image_sim.sh     # (Estimated time: 1 minute)
```

Results saved to `main_focus.csv`, `main_dense.csv`, `main_adaptiv.csv`

### Ablation Study

Simulate *Focus* with only SEC and without SIC:
```bash
python main.py \
    --all_models_datasets \
    --accelerator focus \
    --SEC_only \
    --trace_dir TRACE_DIR \
    --output_dir OUTPUT_DIR     # (Estimated time: 3 minutes)
```

Results saved to `main_focus_SEC_only.csv`

### Worst Case Analysis (Figure 13)

The input tile size in *Focus* varies. We analyze the utilization of *Focus* architecture under different cases:
```bash
python utils/analysis.py --trace_dir TRACE_DIR --output_dir OUTPUT_DIR    # (Estimated time: 10 seconds)
```

Results saved to `figure_13.svg`

---

We provide jupyter notebook scripts in `evaluation_scripts/` to organize and visualize the results of simulation