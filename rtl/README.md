# Focus and Baselines Hardware Implementation

This directory contains RTL (Register Transfer Level) implementations in Verilog for the Focus architecture and baseline designs. These hardware modules are synthesized using Synopsys Design Compiler (DC) for area and power evaluation.

## Focus Architecture Hierarchy
The RTL part of Focus Architecture mainly consists of a systolic array, similarity concentrator (SIC), and semantic concentrator (SEC), and special function unit (SFU)
```
├── Systolic Array (traditional_systolic.v)
│   └── traditional_mac
│       ├── fp16_mult
│       │   ├── gMultiplier (internal)
│       │   └── multiplication_normaliser (internal)
│       ├── fp16_to_fp32
│       ├── fp32_add
│       │   ├── generalAdder (internal)
│       │   └── addition_normaliser (internal)
│       └── fp32_to_fp16
│
│
├── SIC Cosine Similarity (cosine_similarity_unit.sv)
│   ├── fp16_mult
│   │   ├── gMultiplier (internal)
│   │   └── multiplication_normaliser (internal)
│   └── fp16_add
│
├── SIC L2 Norm (inv_magnitude_unit.v)
│   ├── fp16_mult
│   │   ├── gMultiplier (internal)
│   │   └── multiplication_normaliser (internal)
│   ├── fp16_add
│   └── fast_inv_sqrt (internal module in same file)
│       ├── fp16_mult
│       │   ├── gMultiplier (internal)
│       │   └── multiplication_normaliser (internal)
│       └── fp16_add
│
├── SIC Max Unit (max_unit_fp16.sv)
│   └── [No dependencies - uses internal fp16_lt_f function]
│
├── SIC Average Update (average_update_unit.sv)
│   ├── fp16_mult
│   │   ├── gMultiplier (internal)
│   │   └── multiplication_normaliser (internal)
│   └── fp16_add
│
├── SIC Accumulator (fp32_add.v)
│   └── [No dependencies - pure combinational logic]
│
├── SEC Max Unit (max_unit_fp16.sv)
│   └── [No dependencies - uses internal fp16_lt_f function]
|
├── SFU SQRT (fastinvsqrt_fp32.v)
│   ├── fp32_mult
│   │   └── fp32_multiplication_normaliser (internal)
│   └── fp32_add
│       ├── generalAdder (internal)
│       └── addition_normaliser (internal)
│
├── SFU EXP (fp16_exp.v)
│   ├── fp16_mult
│   │   ├── gMultiplier (internal)
│   │   └── multiplication_normaliser (internal)
│   └── fp16_add
│
├── SFU RECIP (fp16_recip.v)
│   ├── fp16_mult (2 instances: u_mult1, u_mult2)
│   │   ├── gMultiplier (internal)
│   │   └── multiplication_normaliser (internal)
│   └── fp16_add
│
├── SFU MULT (fp16_mult.v)
│   ├── gMultiplier (internal)
│   └── multiplication_normaliser (internal)
│
├── SFU ADD (fp16_add.v)
│   └── [No dependencies - pure combinational logic]
```

## Baseline Architecture 

The core logic of baselines are also implemented in RTL. All baselines is equipped with the same SFU as Focus.

Systolic Array
```
├── Systolic Array (traditional_systolic.v)
│   └── traditional_mac
│       ├── fp16_mult
│       │   ├── gMultiplier (internal)
│       │   └── multiplication_normaliser (internal)
│       ├── fp16_to_fp32
│       ├── fp32_add
│       │   ├── generalAdder (internal)
│       │   └── addition_normaliser (internal)
│       └── fp32_to_fp16
```

AdapTiV
```
├── Adaptiv PE Array (adaptiv_array.v)
│   └── traditional_mac
│       ├── fp16_mult
│       │   ├── gMultiplier (internal)
│       │   └── multiplication_normaliser (internal)
│       ├── fp16_to_fp32
│       ├── fp32_add
│       │   ├── generalAdder (internal)
│       │   └── addition_normaliser (internal)
│       └── fp32_to_fp16
```

CMC
```
├── Systolic Array (traditional_systolic.v)
│   └── traditional_mac
│       ├── fp16_mult
│       │   ├── gMultiplier (internal)
│       │   └── multiplication_normaliser (internal)
│       ├── fp16_to_fp32
│       ├── fp32_add
│       │   ├── generalAdder (internal)
│       │   └── addition_normaliser (internal)
│       └── fp32_to_fp16
│
├── CMC Codec Unit (cmc_codec_pe.v)
│   └── adder_tree_64 (internal module in same file)
│
├── CMC Codec Adder Tree (cmc_addertree_4to1.v)
│   └── [No dependencies - pure combinational logic]
```