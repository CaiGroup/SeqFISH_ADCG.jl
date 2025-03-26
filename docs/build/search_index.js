var documenterSearchIndex = {"docs":
[{"location":"api_reference.html#API-Reference","page":"API Reference","title":"API Reference","text":"","category":"section"},{"location":"api_reference.html#2D-Functions","page":"API Reference","title":"2D Functions","text":"","category":"section"},{"location":"api_reference.html","page":"API Reference","title":"API Reference","text":"fit_img_tiles\nfit_2048x2048_img_tiles\nremove_duplicates\nfit_tile","category":"page"},{"location":"api_reference.html#SeqFISH_ADCG.fit_img_tiles","page":"API Reference","title":"SeqFISH_ADCG.fit_img_tiles","text":"fit_img_tiles(img,\n              main_tile_width :: Int64,\n              tile_overlap :: Int64,\n              sigma_lb :: Float64,\n              sigma_ub :: Float64,\n              tau :: Float64,\n              final_loss_improvement :: Float64,\n              min_weight :: Float64,\n              max_iters :: Int64,\n              max_cd_iters :: Int64,\n              noise_mean :: Float64, \n              fit_alg :: AbstractString = \"ADCG\"\n            )\n\nRun ADCG on a square image of arbitrar sized pixel image by breaking it up into overlapping tiles of user specified width and overlap, then running ADCG on each tile and aggregating the results It is necessary to call remove duplicates on the resultant image to remove the duplicates in the regions of overlapping tiles.\n\nArguments:\n\nimg : a 2048x2048 image to fit\nmain_tile_width : the width of the main tile \ntile_overlap : width of the overlaps of tiles with their neighbors \nsigma_lb : the lowest allowed σ of a PSF\nsigma_ub : the highest allowed σ of a PSF\ntau : not used in current version\nfinal_loss_improvement : ADCG terminates when the improvement in the loss function in subsequent iterations is less than this\nmin_weight : ADCG terminates when the next best PSF to add to the model has weight less than this\nmax_cd_iters : the maximum number of iterations of gradient descent to run after adding a PSF to the model to adjust the parameters of all PSFs in the model\nnoise_mean : the noise mean is subtracted from the image before fitting\nfit_alg : \n\nFits gaussian point spread functions in an arbitrarily sizedsquare image with ADCG by splitting it into overlapping pixel tiles of user specified size. \n\n\n\n\n\n","category":"function"},{"location":"api_reference.html#SeqFISH_ADCG.fit_2048x2048_img_tiles","page":"API Reference","title":"SeqFISH_ADCG.fit_2048x2048_img_tiles","text":"fit_2048x2048_img_tiles(img,\n                        sigma_lb :: Float64,\n                        sigma_ub :: Float64,\n                        tau :: Float64,\n                        final_loss_improvement :: Float64,\n                        min_weight :: Float64,\n                        max_iters :: Int64,\n                        max_cd_iters :: Int64,\n                        noise_mean :: Float64\n                        )\n\nArguments:\n\nimg : a 2048x2048 image to fit \nsigma_lb : the lowest allowed σ of a PSF\nsigma_ub : the highest allowed σ of a PSF\ntau : not used in current version\nfinal_loss_improvement : ADCG terminates when the improvement in the loss function in subsequent iterations is less than this\nmin_weight : ADCG terminates when the next best PSF to add to the model has weight less than this\nmax_cd_iters : the maximum number of iterations of gradient descent to run after adding a PSF to the model to adjust the parameters of all PSFs in the model\nnoise_mean : the noise mean is subtracted from the image before fitting\n\nFits gaussian point spread functions in a 2048x2048 pixel image with ADCG by splitting it into overlapping 70x70 pixel tiles. \n\n\n\n\n\n","category":"function"},{"location":"api_reference.html#SeqFISH_ADCG.remove_duplicates","page":"API Reference","title":"SeqFISH_ADCG.remove_duplicates","text":"remove_duplicates(points :: DataFrame,\n                img,\n                sigma_lb :: Float64,\n                sigma_ub :: Float64,\n                tau :: Float64,\n                noise_mean :: Float64,\n                min_allowed_separation :: Float64,\n                dims :: Int64 = 2\n                )\n\nRemoves duplicates within min_allowed_separtion of each other from an image.\nThis is necessary when fit by a tiled ADCG where the tiles overlap.\n\n\n\n\n\n","category":"function"},{"location":"api_reference.html#SeqFISH_ADCG.fit_tile","page":"API Reference","title":"SeqFISH_ADCG.fit_tile","text":"fit_tile(inputs)\n\nArguments:\n\ninputs : A tuple of inputs with entries:\ntile : An image tile to perform ADCG on.\nsigma_lb : the lowest allowed value of the width parameter of dots.\nsigma_ub : the highest allowed value of the width parameter of dots.\nnoise_mean : estimated mean of noise. Pixel values below this are set to zero.\ntau : not used in this version\nfinal_loss_improvement : Terminate ADCG when the objective improves by less than this value in one iteration\nmin_weight : The minimum allowed weight of a PSF in the image model\nmax_iters : The maximum number of ADCG iterations, or number PSFs to add to the model.\nmax_cd_iterations : the maximum number of times to perform gradient descent for the parameter values of all dots.\n\n\n\n\n\n","category":"function"},{"location":"api_reference.html#3D-Functions","page":"API Reference","title":"3D Functions","text":"","category":"section"},{"location":"api_reference.html","page":"API Reference","title":"API Reference","text":"fit_stack_tiles\nfit_stack\nremove_duplicates3d\nremove_duplicates_ignore_z","category":"page"},{"location":"api_reference.html#SeqFISH_ADCG.fit_stack_tiles","page":"API Reference","title":"SeqFISH_ADCG.fit_stack_tiles","text":"fit_stack_tiles(\n                   img_stack,\n                   main_tile_width :: Int64,\n                   tile_overlap :: Int64,\n                   tile_depth :: Int64,\n                   tile_depth_overhang :: Int64,\n                   sigma_xy_lb :: Float64,\n                   sigma_xy_ub :: Float64,\n                   sigma_z_lb :: Float64,\n                   sigma_z_ub :: Float64,\n                   final_loss_improvement :: Float64,\n                   min_weight :: Float64,\n                   max_iters :: Int64,\n                   max_cd_iters :: Int64,\n                   fit_alg = \"ADCG\"\n            )\n\nRun ADCG on a square image of arbitrar sized pixel image by breaking it up into overlapping tiles of user specified width and overlap, then running ADCG on each tile and aggregating the results It is necessary to call remove duplicates on the resultant image to remove the duplicates in the regions of overlapping tiles.\n\nArguments:\n\nimg : a 2048x2048 image to fit\nmain_tile_width : the width of the main tile \ntile_overlap : width of the overlaps of tiles with their neighbors \nsigma_xy_lb : the lowest allowed lateral σ of a PSF\nsigma_xy_ub : the highest allowed lateral σ of a PSF\nsigma_z_lb : the lowest allowed axial σ of a PSF\nsigma_z_ub : the highest allowed axial σ of a PSF\nfinal_loss_improvement : ADCG terminates when the improvement in the loss function in subsequent iterations is less than this\nmin_weight : ADCG terminates when the next best PSF to add to the model has weight less than this\nmax_iters : the maximum number of iterations in which a new PSF may be added to the model of a tile (i.e. the maximum PSFs in a tile)\nmax_cd_iters : the maximum number of iterations of gradient descent to run after adding a PSF to the model to adjust the parameters of all PSFs in the model\nnoise_mean : the noise mean is subtracted from the image before fitting\n\nFits gaussian point spread functions in an arbitrarily sizedsquare image with ADCG by splitting it into overlapping pixel tiles of user specified size. \n\n\n\n\n\n","category":"function"},{"location":"api_reference.html#SeqFISH_ADCG.fit_stack","page":"API Reference","title":"SeqFISH_ADCG.fit_stack","text":"fit_tile(inputs)\n\nArguments:\n\ninputs : A tuple of inputs with entries:\ntile : An image tile to perform ADCG on.\nsigma_lb : the lowest allowed value of the width parameter of dots.\nsigma_ub : the highest allowed value of the width parameter of dots.\nnoise_mean : estimated mean of noise. Pixel values below this are set to zero.\ntau : not used in this version\nfinal_loss_improvement : Terminate ADCG when the objective improves by less than this value in one iteration\nmin_weight : The minimum allowed weight of a PSF in the image model\nmax_iters : The maximum number of ADCG iterations, or number PSFs to add to the model.\nmax_cd_iterations : the maximum number of times to perform gradient descent for the parameter values of all dots.\n\n\n\n\n\n","category":"function"},{"location":"api_reference.html#SeqFISH_ADCG.remove_duplicates3d","page":"API Reference","title":"SeqFISH_ADCG.remove_duplicates3d","text":"remove_duplicates3d(points :: DataFrame,\n                img,\n                sigma_lb :: Float64,\n                sigma_ub :: Float64,\n                min_allowed_separation :: Float64,\n                )\n\nRemoves duplicates within min_allowed_separtion (voxels) of each other from an image.\nThis is necessary when fit by a tiled ADCG where the tiles overlap.\n\n\n\n\n\n","category":"function"},{"location":"api_reference.html#SeqFISH_ADCG.remove_duplicates_ignore_z","page":"API Reference","title":"SeqFISH_ADCG.remove_duplicates_ignore_z","text":"remove_duplicates_ignore_z(points :: DataFrame,\n                img,\n                sigma_lb :: Float64,\n                sigma_ub :: Float64,\n                tau :: Float64,\n                noise_mean :: Float64,\n                min_allowed_separation :: Float64,\n                dims :: Int64 = 2\n                )\n\nRemoves duplicates within min_allowed_separtion of each other from an image in the xy directions.\nIn cases where z fitting is unreliable, it may be best to ingore the z coordinate for determining duplicates.\nThis is necessary when fit by a tiled ADCG where the tiles overlap.\n\n\n\n\n\n","category":"function"},{"location":"installation.html#Installation","page":"Installation","title":"Installation","text":"","category":"section"},{"location":"installation.html","page":"Installation","title":"Installation","text":"From the Julia REPL, install by running the commands:","category":"page"},{"location":"installation.html","page":"Installation","title":"Installation","text":"julia> using Pkg\njulia> Pkg.add(url=\"https://github.com/CaiLab/SeqFISH_ADCG\")","category":"page"},{"location":"installation.html","page":"Installation","title":"Installation","text":"Alternatively, open the package manager by typing","category":"page"},{"location":"installation.html","page":"Installation","title":"Installation","text":"julia>] ","category":"page"},{"location":"installation.html","page":"Installation","title":"Installation","text":"then","category":"page"},{"location":"installation.html","page":"Installation","title":"Installation","text":"Pkg> add \"https://github.com/CaiGroup/SeqFISHSyndromeDecoding\"","category":"page"},{"location":"example_FitDots.html#Example","page":"Example","title":"Example","text":"","category":"section"},{"location":"example_FitDots.html","page":"Example","title":"Example","text":"using Plots\nusing SeqFISH_ADCG\nusing FileIO\nusing Images","category":"page"},{"location":"example_FitDots.html","page":"Example","title":"Example","text":"Load example data","category":"page"},{"location":"example_FitDots.html","page":"Example","title":"Example","text":"ro_img = load(\"example_data/ro_preprocessed.png\")\nro_img = reinterpret.(UInt16, channelview(ro_img));","category":"page"},{"location":"example_FitDots.html","page":"Example","title":"Example","text":"Set Parameters","category":"page"},{"location":"example_FitDots.html","page":"Example","title":"Example","text":"sigma_lb = 0.87\nsigma_ub = 1.22\ntau = 2.0*10^12\nfinal_loss_improvement = 1000.0\nmin_weight = 800.0\nmax_iters = 200\nmax_cd_iters = 20\nthreshold = 0.0;","category":"page"},{"location":"example_FitDots.html","page":"Example","title":"Example","text":"For expediancy of the demonstration, we will choose a small example tile to run ADCG on","category":"page"},{"location":"example_FitDots.html","page":"Example","title":"Example","text":"example_tile = ro_img[1020:1080, 1220:1280]\nheatmap(example_tile)","category":"page"},{"location":"example_FitDots.html","page":"Example","title":"Example","text":"(Image: svg)","category":"page"},{"location":"example_FitDots.html","page":"Example","title":"Example","text":"Now we run ADCG on the tile sample tile","category":"page"},{"location":"example_FitDots.html","page":"Example","title":"Example","text":"inputs = (example_tile, sigma_lb, sigma_ub, threshold, tau, final_loss_improvement, min_weight, max_iters, max_cd_iters)\nps = fit_tile(inputs)\n\nheatmap(example_tile)\nscatter!(ps[1,:], ps[2,:])","category":"page"},{"location":"example_FitDots.html","page":"Example","title":"Example","text":"(Image: svg)","category":"page"},{"location":"example_FitDots.html","page":"Example","title":"Example","text":"ADCG appears to pick up all of the dots. It may have a few extra, but it better to tune the parameters such that too many dots are picked up than too few because SeqFISHSyndromeDecoding is very effective at discarding bad dots.","category":"page"},{"location":"example_FitDots.html","page":"Example","title":"Example","text":"Running ADCG on a whole image requires breaking up the image into overlapping tiles, running ADCG on each tile, and piecing the tiles back together. All of the Cai Lab's microscopes use 2048X2048 cameras, so SeqFISH_ADCG comes with a special function, fit_2048x2048_img_tiles, that breaks 2048 images up into tiles","category":"page"},{"location":"example_FitDots.html","page":"Example","title":"Example","text":"points_with_duplicates = fit_2048x2048_img_tiles(ro_img, sigma_lb, sigma_ub, tau, final_loss_improvement, min_weight, max_iters, max_cd_iters, threshold)","category":"page"},{"location":"example_FitDots.html","page":"Example","title":"Example","text":"If you have images that are not of 2048x2048 pixels, you will need to use fit_img_tiles, which fit_2048x2048_img_tiles wraps, and specify your own tile and overlap size. For example fit_2048x2048_img_tiles calls:","category":"page"},{"location":"example_FitDots.html","page":"Example","title":"Example","text":"tile_width = 64\ntile_overlap = 6\n\npoints_with_duplicates2 = fit_2048x2048_img_tiles(ro_img, tile_width, tile_overlap, sigma_lb, sigma_ub, tau, final_loss_improvement, min_weight, max_iters, max_cd_iters, threshold)","category":"page"},{"location":"example_FitDots.html","page":"Example","title":"Example","text":"You will need to ensure that the width and height of your image is divisible by tile_width.","category":"page"},{"location":"example_FitDots.html","page":"Example","title":"Example","text":"The next step is that you will need to remove dots that are too close to each other. This removes duplicates that are in the overlapping regions of the tiles, or may have just been fit twice by ADCG","category":"page"},{"location":"example_FitDots.html","page":"Example","title":"Example","text":"min_allowed_separation=2.0\npoints = remove_duplicates(points_with_duplicates, rp+img, sigma_lb, sigma_ub, tau, noise_mean, min_allowed_separation)","category":"page"},{"location":"example_FitDots.html","page":"Example","title":"Example","text":"CSV.write(\"example_fit.csv\", points)","category":"page"},{"location":"index.html#SeqFISH_ADCG.jl","page":"SeqFISH_ADCG.jl","title":"SeqFISH_ADCG.jl","text":"","category":"section"},{"location":"index.html#Introduction","page":"SeqFISH_ADCG.jl","title":"Introduction","text":"","category":"section"},{"location":"index.html","page":"SeqFISH_ADCG.jl","title":"SeqFISH_ADCG.jl","text":"SeqFISH_ADCG (Sequential Fluoresceint Insitue Hybridization Alternating Descent Conditional Gradient) is adapted version of SparseInverseProblems.jl which implements the ADCG algorithm originally described in this paper. A benchmarking paper showed that ADCG excels at fitting single molecule microscopy images.","category":"page"},{"location":"index.html","page":"SeqFISH_ADCG.jl","title":"SeqFISH_ADCG.jl","text":"ADCG does this by taking an alternating approach to fitting: add a single molecule to the model, then adjust the entire, repeat.","category":"page"},{"location":"index.html","page":"SeqFISH_ADCG.jl","title":"SeqFISH_ADCG.jl","text":"Here, we extend ADCG to apply it specifically to SeqFISH Data. ","category":"page"},{"location":"index.html","page":"SeqFISH_ADCG.jl","title":"SeqFISH_ADCG.jl","text":"Generally, this package can be used to fit images that are composed of a superposition of Gaussian point spread functions. The fit procedure is iterative. At each iteration, a new point spread function is added to the model of the image, and the parameters (position and width) of the point spread functions previously in the model are adjusted to accomodate the newcomer.","category":"page"},{"location":"index.html","page":"SeqFISH_ADCG.jl","title":"SeqFISH_ADCG.jl","text":"The example jupyter notebooks show how to preprocess a SeqFISH image, and then how use the package to fit a model locating the point spread functions.","category":"page"},{"location":"index.html#Contents","page":"SeqFISH_ADCG.jl","title":"Contents","text":"","category":"section"},{"location":"index.html","page":"SeqFISH_ADCG.jl","title":"SeqFISH_ADCG.jl","text":"Pages = [\"installation.md\", \"example_FitDots.md\", \"api_reference.md\"]\nDepth = 3","category":"page"}]
}
