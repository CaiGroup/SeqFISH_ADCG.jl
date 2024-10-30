export fit_stack_tiles, fit_stack, remove_duplicates3d, remove_duplicates_ignore_z

using DataFrames

function fit_stack(inputs)
    #println("fitting tile ... ")
    (stack, sigma_xy_lb, sigma_xy_ub, sigma_z_lb, sigma_z_ub, final_loss_improvement, min_weight, max_iters, max_cd_iters, fit_alg) = inputs
    n_pixels = maximum(size(stack)[1:2])
    n_slices = size(stack)[3]
    #min_weight = 200.0
    if any(size(stack)[1:2] .< n_pixels)
        orig_stack = copy(stack)
        stack = zeros(n_pixels, n_pixels, n_slices)
        stack[1:size(orig_stack)[1], 1:size(orig_stack)[2], :] = orig_stack
    end
    grid = n_pixels*2

    gb_sim = GaussBlur3D(sigma_xy_lb, sigma_xy_ub, sigma_z_lb, sigma_z_ub, n_pixels, n_slices)
    #GaussBlur3D(sigma_lb, sigma_ub, n_pixels, grid)

    target = Float64.(vec(stack)) #.-noise_mean#*1.5
    target[target .< 0] .= 0

    
    sum_target = sum(target)
    """
    if sum_target == 0
        return [], []
    end
    """

    #tau = sum(target[target .> 0])/final_loss_improvement + 1.0#/5.0 + 1.0
    #tau = sum_target
    tau = sum_target/15 + 1.0 #final_loss_improvement + 1.0
    #println("tau: ", tau)
    function callback(old_thetas,thetas,output,old_obj_val)
      #evalute current OV
      new_obj_val,t = loss(LSLoss(), output - target)
      if old_obj_val - new_obj_val < final_loss_improvement
        return true
      end
      return false
    end
    records = run_fit(gb_sim, LSLoss(), target, tau, min_weight, max_iters=max_iters, callback=callback, max_cd_iters=max_cd_iters,fit_alg=fit_alg)
    return records
end


function fit_stack_tiles(img_stack,
                       main_tile_width :: Int64,
                       tile_overlap :: Int64,
                       tile_depth :: Int64,
                       tile_depth_overhang :: Int64,
                       sigma_xy_lb :: Float64,
                       sigma_xy_ub :: Float64,
                       sigma_z_lb :: Float64,
                       sigma_z_ub :: Float64,
                       final_loss_improvement :: Float64,
                       min_weight :: Float64,
                       max_iters :: Int64,
                       max_cd_iters :: Int64,
                       fit_alg = "ADCG"
                )

    #tile_depth = 3
    #tile_depth_overhang= 1

    #@assert size(img) == (2048, 2048)
    img_height, img_width, nslices = size(img_stack)

    tile_width = main_tile_width#32
    overhang = tile_overlap

    #tiles_across = Int64(2048/tile_width)
    tiles_across = ceil(Int64, img_width/tile_width)
    tiles_deep = ceil(Int64, nslices/tile_depth)
    bnds_start = tile_width*Array(0:(tiles_across-1)) .+ 1
    bnds_end = tile_width*Array(1:(tiles_across-1)) .+ overhang
    z_start = tile_depth*Array(0:(tiles_deep-1)) .+ 1
    z_end = tile_depth*Array(1:(tiles_deep-1)) .+ tile_depth_overhang
    println("img_width; $img_width")
    println("tile_width: $tile_width")
    println("tiles_across: $tiles_across")

    push!(bnds_end, img_width)
    push!(z_end, nslices)

    # define overlapping tiles
    coords = [(bnds_start[i],bnds_end[i], bnds_start[j],bnds_end[j], z_start[k], z_end[k]) for i in 1:tiles_across, j in 1:tiles_across, k in 1:tiles_deep]
    stacks = [img_stack[cds[1]:cds[2], cds[3]:cds[4], cds[5]:cds[6]] for cds in coords]
    fit_tile_inputs = [(stacks[i], sigma_xy_lb, sigma_xy_ub, sigma_z_lb, sigma_z_ub, final_loss_improvement, min_weight, max_iters, max_cd_iters, fit_alg) for i in 1:length(stacks)]
    #fit_tile_inputs = [(tiles[i], sigma_lb, sigma_ub, noise_mean, tau, final_loss_improvement, min_weight, max_iters, max_cd_iters) for i in 1:length(tiles)]

    #fit tiles
    println("length(fit_tile_inputs): ", length(fit_tile_inputs))
    tile_fits = map(fit_stack, fit_tile_inputs)
    #tile_fits = map(fit_tile, fit_tile_inputs)
    #println("finished fitting tiles, putting together now...")

    #throw out points within 2 pixels of the edge of each tile.
    #trimmed_tile_fits = [trim_tile_fit!(tile_fit, img_width + tile_overlap) for tile_fit in tile_fits]
    trimmed_tile_fits = [trim_tile_records!(tile_fit, tile_width + tile_overlap) for tile_fit in tile_fits]

    #concatenate points

    final_points, record = tiles_to_img_stack(trimmed_tile_fits, coords)


    return (final_points[:,1:6], record)

    #return points_df
end

function tiles_to_img_stack(trimmed_tile_fits, coords)
    ps_final = [trimmed_tile_fits[1].last_iteration[1:0, :]]
    ps_records = [trimmed_tile_fits[1].records[1:0, :]]
    for i = 1:length(trimmed_tile_fits)
        p = trimmed_tile_fits[i]

        #p .*= (tile_width + overhang)
        if nrow(p.last_iteration) > 0
            p.last_iteration[:,"x"] .+= coords[i][3] .- 1
            p.last_iteration[:,"y"] .+= coords[i][1] .- 1
            p.last_iteration[:,"z"] .+= coords[i][5] .- 1
            p.records[:,"x"] .+= coords[i][3] .- 1
            p.records[:,"y"] .+= coords[i][1] .- 1
            p.records[:,"z"] .+= coords[i][5] .- 1
            push!(ps_final, p.last_iteration)
            push!(ps_records, p.records)

            #p[1,:] .+= coords[i][3]
            #p[2,:] .+= coords[i][1]
            #p[3,:] .+= coords[i][5]
            #push!(ps, p)#trimmed_tile_fits[i][1])
        end
    end
    return vcat(ps_final...), vcat(ps_records...)
    #println("length(ps): ", length(ps))
    #ps = hcat(ps...)
    #println("size(ps): ", size(ps))

    #points_df = DataFrame()
    #points_df[!, "x"] = ps[1,:]
    #points_df[!, "y"] = ps[2,:]
    #points_df[!, "z"] = ps[3,:]
    #points_df[!, "s_xy"] = ps[4,:]
    #points_df[!, "s_z"] = ps[5,:]
    #points_df[!, "w"] = ps[6,:]
end

function remove_duplicates3d(points :: DataFrame,
                           sigma_lb :: Float64,
                           sigma_ub :: Float64,
                           min_allowed_separation :: Float64
                           )
    if nrow(points) == 0
        return points
    end
    
    ps = Array(hcat(points[!, "x"],
                   points[!, "y"],
                   points[!,"z"],
                   points[!, "sxy"],
                   points[!, "sz"],
                   points[!, "w"])')

    while true
        ps_new = _remove_duplicates(ps, zeros(2,2), sigma_lb, sigma_ub, 1.0, 0.0, min_allowed_separation, 3)
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
    points_df[!, "z"] = ps[3,:]
    points_df[!, "sxy"] = ps[4,:]
    points_df[!, "sz"] = ps[5,:]
    points_df[!, "w"] = ps[6,:]
    return points_df
end

function remove_duplicates_ignore_z(points :: DataFrame,
    sigma_lb :: Float64,
    sigma_ub :: Float64,
    min_allowed_separation :: Float64
    )

    if nrow(points) == 0
        return points
    end
    
    ps = Array(hcat(points[!, "x"],
                   points[!, "y"],
                   points[!, "sxy"],
                   points[!, "w"])')

    while true
        ps_new = _remove_duplicates(ps, zeros(2,2), sigma_lb, sigma_ub, 1.0, 0.0, min_allowed_separation, 2)
        if prod(size(ps)) == prod(size(ps_new))
            ps = ps_new
            break
        else
            ps = ps_new
        end
    end

    new_points = DataFrame(ps', ["x","y","sxy","w"])
    points_df = innerjoin(points, new_points,on=["x","y","sxy","w"])

    return points_df
end