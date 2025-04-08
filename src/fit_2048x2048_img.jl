# Author: Jonathan A. White
# Date Created: February 10, 2021

export fit_img_tiles, fit_2048x2048_img_tiles, fit_tile, remove_duplicates

using DataFrames
using Statistics
using NearestNeighbors

"""
    fit_tile(inputs)

Arguments:
- `inputs` : A tuple of inputs with entries:
    - `tile` : An image tile to perform ADCG on.
    - `sigma_lb` : the lowest allowed value of the width parameter of dots.
    - `sigma_ub` : the highest allowed value of the width parameter of dots.
    - `noise_mean` : estimated mean of noise. Pixel values below this are set to zero.
    - `tau` : not used in this version
    - `final_loss_improvement` : Terminate ADCG when the objective improves by less than this value in one iteration
    - `min_weight` : The minimum allowed weight of a PSF in the image model
    - `max_iters` : The maximum number of ADCG iterations, or number PSFs to add to the model.
    - `max_cd_iterations` : the maximum number of times to perform gradient descent for the parameter values of all dots.
    - `fit_alg` : 'ADCG' or 'DAO', uses the respective algorithm. Only ADCG is thorougly tested.
"""
function fit_tile(inputs)
    tile, sigma_lb, sigma_ub, noise_mean, tau, final_loss_improvement, min_weight, max_iters, max_cd_iters, fit_alg = inputs

    n_pixels = maximum(size(tile))
    if any(size(tile) .< n_pixels)
        orig_tile = copy(tile)
        tile = zeros(n_pixels, n_pixels)
        tile[1:size(orig_tile)[1], 1:size(orig_tile)[2]] = orig_tile
    end
    grid = n_pixels*2

    gb_sim = GaussBlur2D(sigma_lb, sigma_ub, n_pixels)

    target = vec(tile).-noise_mean
    target[target .< 0] .= 0

    sum_target = sum(target)
    if sum_target == 0
        return initialize_dot_records(gb_sim, [[] []]) #[]
    end

    #tau = sum(target[target .> 0])/final_loss_improvement + 1.0#/5.0 + 1.0
    #tau = sum_target
    tau = sum_target/15 + 1.0 #final_loss_improvement + 1.0
    function callback(old_thetas,thetas,output,old_obj_val)
      #evalute current OV
      new_obj_val,t = loss(LSLoss(), output - target)
      #println("new_obj_val: ", new_obj_val)
      #println("old_obj_val: ", old_obj_val)
      if old_obj_val - new_obj_val < final_loss_improvement
        #println("old_obj: $old_obj_val")
        #println("new_obj: $new_obj_val")
        return true
      end
      return false
    end
    
    return run_fit(gb_sim, LSLoss(), target, tau, min_weight, max_iters=max_iters, callback=callback, max_cd_iters=max_cd_iters,fit_alg=fit_alg)
end

function trim_tile_records!(tile_records :: DotRecords, width :: Int64)
    trim_tile_fit!(tile_records.records, width)
    trim_tile_fit!(tile_records.last_iteration, width)
    return tile_records
end

function trim_tile_fit!(tile_fit :: DataFrame, width :: Int64)
    ps = tile_fit
    if nrow(ps) == 0
        return tile_fit
    end


    to_keep = (ps.x .> 2) .| (ps.y .> 2) .| (ps.x .< width-2) .| (ps.y .< width-2)
    return ps[to_keep, :]
end

"""
    fit_2048x2048_img_tiles(img,
                            sigma_lb :: Float64,
                            sigma_ub :: Float64,
                            tau :: Float64,
                            final_loss_improvement :: Float64,
                            min_weight :: Float64,
                            max_iters :: Int64,
                            max_cd_iters :: Int64,
                            noise_mean :: Float64,
                            fit_alg :: AbstractString = "ADCG"
                            )

Arguments:
- `img` : a 2048x2048 image to fit 
- `sigma_lb` : the lowest allowed σ of a PSF
- `sigma_ub` : the highest allowed σ of a PSF
- `tau` : not used in current version
- `final_loss_improvement` : ADCG terminates when the improvement in the loss function in subsequent iterations is less than this
- `min_weight` : ADCG terminates when the next best PSF to add to the model has weight less than this
- `max_cd_iters` : the maximum number of iterations of gradient descent to run after adding a PSF to the model to adjust the parameters of all PSFs in the model
- `noise_mean` : the noise mean is subtracted from the image before fitting
- `fit_alg` : 'ADCG' or 'DAO', uses the respective algorithm. Only ADCG is thorougly tested. Must be set as a keyword argument.

Fits gaussian point spread functions in a 2048x2048 pixel image with ADCG by splitting it into overlapping 70x70 pixel tiles. 

"""
function fit_2048x2048_img_tiles(img,
                           sigma_lb :: Float64,
                           sigma_ub :: Float64,
                           tau :: Float64,
                           final_loss_improvement :: Float64,
                           min_weight :: Float64,
                           max_iters :: Int64,
                           max_cd_iters :: Int64,
                           noise_mean :: Float64;
                           fit_alg :: AbstractString = "ADCG"
        )
    @assert size(img) == (2048, 2048)
    fit_img_tiles(img, 64, 6, sigma_lb, sigma_ub, tau, final_loss_improvement,
                               min_weight,
                               max_iters,
                               max_cd_iters,
                               noise_mean,
                               fit_alg = fit_alg
                )
end

"""
    fit_img_tiles(img,
                  main_tile_width :: Int64,
                  tile_overlap :: Int64,
                  sigma_lb :: Float64,
                  sigma_ub :: Float64,
                  tau :: Float64,
                  final_loss_improvement :: Float64,
                  min_weight :: Float64,
                  max_iters :: Int64,
                  max_cd_iters :: Int64,
                  noise_mean :: Float64,
                  fit_alg :: AbstractString = "ADCG"
                )
Run ADCG on a square image of arbitrar sized pixel image by breaking it up into overlapping tiles of user specified
width and overlap, then running ADCG on each tile and aggregating the results
It is necessary to call remove duplicates on the resultant image to remove the duplicates in the regions of overlapping tiles.

Arguments:
- `img` : a 2048x2048 image to fit
- `main_tile_width` : the width of the main tile 
- `tile_overlap` : width of the overlaps of tiles with their neighbors 
- `sigma_lb` : the lowest allowed σ of a PSF
- `sigma_ub` : the highest allowed σ of a PSF
- `tau` : not used in current version
- `final_loss_improvement` : ADCG terminates when the improvement in the loss function in subsequent iterations is less than this
- `min_weight` : ADCG terminates when the next best PSF to add to the model has weight less than this
- `max_cd_iters` : the maximum number of iterations of gradient descent to run after adding a PSF to the model to adjust the parameters of all PSFs in the model
- `noise_mean` : the noise mean is subtracted from the image before fitting
- `fit_alg` : 'ADCG' or 'DAO', uses the respective algorithm. Only ADCG is thorougly tested. Must be set as a keyword argument.

Fits gaussian point spread functions in an arbitrarily sizedsquare image with ADCG by splitting it into overlapping pixel tiles of user specified size. 

"""
function fit_img_tiles(img,
                       main_tile_width :: Int64,
                       tile_overlap :: Int64,
                       sigma_lb :: Float64,
                       sigma_ub :: Float64,
                       tau :: Float64,
                       final_loss_improvement :: Float64,
                       min_weight :: Float64,
                       max_iters :: Int64,
                       max_cd_iters :: Int64,
                       noise_mean :: Float64;
                       fit_alg :: AbstractString = "ADCG"
        )

    img_height, img_width = size(img)

    tile_width = main_tile_width
    overhang = tile_overlap

    tiles_across = ceil(Int64, img_width/tile_width)
    bnds_start = tile_width*Array(0:(tiles_across-1)) .+ 1
    bnds_end = tile_width*Array(1:(tiles_across-1)) .+ overhang
    push!(bnds_end, img_width)

    # define overlapping tiles
    coords = [((bnds_start[i]),bnds_end[i], (bnds_start[j]),bnds_end[j]) for i in 1:tiles_across, j in 1:(tiles_across-1)]
    tiles = [img[cds[1]:cds[2],cds[3]:cds[4]] for cds in coords]

    fit_tile_inputs = [(tiles[i], sigma_lb, sigma_ub, noise_mean, tau, final_loss_improvement, min_weight, max_iters, max_cd_iters, fit_alg) for i in 1:length(tiles)]

    #fit tiles
    tile_fits = map(fit_tile, fit_tile_inputs)

    #throw out points within 2 pixels of the edge of each tile.
    trimmed_tile_fits = [trim_tile_records!(tile_fit, img_width + tile_overlap) for tile_fit in tile_fits]
    
    #concatenate points

    final_points, record = tiles_to_img(trimmed_tile_fits, coords)


    return (final_points[:,1:4], record)
end

function tiles_to_img(trimmed_tile_fits, coords)
    ps_final = [trimmed_tile_fits[1].last_iteration[1:0, :]]
    ps_records = [trimmed_tile_fits[1].records[1:0, :]]
    for i = 1:length(trimmed_tile_fits)
        p = trimmed_tile_fits[i]
        if nrow(p.last_iteration) > 0
            p.last_iteration[:,"x"] .+= coords[i][3] .- 1
            p.last_iteration[:,"y"] .+= coords[i][1] .- 1
            p.records[:,"x"] .+= coords[i][3] .- 1
            p.records[:,"y"] .+= coords[i][1] .- 1
            push!(ps_final, p.last_iteration)
            push!(ps_records, p.records)
        end
    end
    return vcat(ps_final...), vcat(ps_records...)
end

"""
    remove_duplicates(points :: DataFrame,
                    img,
                    sigma_lb :: Float64,
                    sigma_ub :: Float64,
                    tau :: Float64,
                    noise_mean :: Float64,
                    min_allowed_separation :: Float64,
                    dims :: Int64 = 2
                    )

    Removes duplicates within min_allowed_separtion of each other from an image.
    This is necessary when fit by a tiled ADCG where the tiles overlap.
"""
function remove_duplicates(points :: DataFrame,
                           img,
                           sigma_lb :: Float64,
                           sigma_ub :: Float64,
                           tau :: Float64,
                           noise_mean :: Float64,
                           min_allowed_separation :: Float64,
                           dims :: Int64 = 2
                           )

    if nrow(points) == 0
        return points
    end
    ps = Matrix(hcat(points[!, "x"], points[!, "y"], points[!, "s"], points[!, "w"])')
    while true
        ps_new = _remove_duplicates(ps, img, sigma_lb, sigma_ub, tau, noise_mean, min_allowed_separation, dims)
        if prod(size(ps)) == prod(size(ps_new))
            ps = ps_new
            break
        else
            ps = ps_new
        end
    end

    points_df = DataFrame()
    points_df[!, "x"] = ps[1,:]
    points_df[!, "y"] = ps[2,:]
    points_df[!, "s"] = ps[3,:]
    points_df[!, "w"] = ps[4,:]
    return points_df
end

function _remove_duplicates(ps :: Matrix,
                           img,
                           sigma_lb :: Float64,
                           sigma_ub :: Float64,
                           tau :: Float64,
                           noise_mean :: Float64,
                           min_allowed_separation :: Float64,
                           dims :: Int64
                           )

    #run solveFiniteDimProblem to remove duplicate points
    coords = ps[1:dims, :]
    target = vec(img).-noise_mean
    target[target .< 0] .= 0
    tau = sum(target)
    bt = BallTree(coords)
    grps = inrange(bt, coords, min_allowed_separation, true)
    lone_point_grps = filter(g->length(g) == 1, grps)
    lone_point_inds = map(ia->ia[1], lone_point_grps)

    lone_points = ps[:, lone_point_inds]

    filter!(g-> length(g)>1, grps)
    grps = unique(grps)

    grp_brightest_inds = []
    for grp in grps
        push!(grp_brightest_inds, grp[argmax(ps[4,grp])])
    end
    sort!(grp_brightest_inds)
    bright_dup_ps = ps[:, grp_brightest_inds]

    ps = hcat(lone_points, bright_dup_ps)
    return ps
end
