# SeqFISH_ADCG.jl

## Introduction

SeqFISH_ADCG (Sequential Fluoresceint Insitue Hybridization Alternating Descent Conditional Gradient) is adapted version of [SparseInverseProblems.jl](https://github.com/nboyd/SparseInverseProblems.jl) which implements the ADCG algorithm originally described in this [paper](https://doi.org/10.1137/15M1035793). A [benchmarking paper](https://doi.org/10.1038/s41592-019-0364-4) showed that ADCG excels at fitting single molecule microscopy images.

ADCG does this by taking an alternating approach to fitting: add a single molecule to the model, then adjust the entire, repeat.

Here we extend ADCG to apply it specifically to SeqFISH Data. 

## Installation

From Julia, install by running the commands:

```
import Pkg
Pkg.add(url="https://github.com/CaiLab/SeqFISH_ADCG")
```


## API Reference

```@docs
fit_img_tiles
fit_2048x2048_img_tiles
remove_duplicates
fit_tile
```