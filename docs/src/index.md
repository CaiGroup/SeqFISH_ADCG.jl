# SeqFISH_ADCG.jl

## Introduction

SeqFISH_ADCG (Sequential Fluoresceint Insitue Hybridization Alternating Descent Conditional Gradient) is adapted version of [SparseInverseProblems.jl](https://github.com/nboyd/SparseInverseProblems.jl) which implements the ADCG algorithm originally described in this [paper](https://doi.org/10.1137/15M1035793). A [benchmarking paper](https://doi.org/10.1038/s41592-019-0364-4) showed that ADCG excels at fitting single molecule microscopy images.

ADCG does this by taking an alternating approach to fitting: add a single molecule to the model, then adjust the entire, repeat.

Here we extend ADCG to apply it specifically to SeqFISH Data. 

Generally, this package can be used to fit images that are composed of a superposition of Gaussian point spread functions. The fit procedure is iterative. At each iteration, a new point spread function is added to the model of the image, and the parameters (position and width) of the point spread functions previously in the model are adjusted to accomodate the newcomer.

The example jupyter notebooks show how to use the package to fit a model locating the point spread functions.


## Contents
```@contents
Pages = ["installation.md", "example_FitDots.md", "api_reference.md"]
Depth = 3
```