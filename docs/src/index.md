# SeqFISH_ADCG.jl

## Introduction

SeqFISH_ADCG is adapted version of [ADCG](https://github.com/nboyd/SparseInverseProblems.jl) which was originally described in this [paper](https://doi.org/10.1137/15M1035793).

ADCG is an algorithm for solving sparse inverse problems that excels at [fitting single molecule microscopy images](https://doi.org/10.1038/s41592-019-0364-4).

Here we extend ADCG to apply it specifically to SeqFISH Data. 


## API Reference

```@docs
fit_img_tiles
fit_2048x2048_img_tiles
remove_duplicates
fit_tile
```