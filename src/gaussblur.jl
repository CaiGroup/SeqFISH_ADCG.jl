export GaussBlur, GaussBlur2D

using LinearAlgebra
using GLM
using DataFrames
using Clustering
#using Debugger

abstract type GaussBlur <: BoxConstrainedDifferentiableModel end

struct GaussBlur2D <: GaussBlur
  sigma_lb :: Float64
  sigma_ub :: Float64
  n_pixels :: Int64
  ng :: Int64
  psf_thresh :: Int64
  grid_f
  grid_sigma
  dims :: Int64
end

function GaussBlur2D(sigma_lb :: Real, sigma_ub :: Real, np :: Int64)
  psf_thresh = ceil(Int64, sigma_ub*3.0)
  grid_f = computeFs(Array(0.5:0.5:np),np, ones(2np).*sigma_lb, psf_thresh)
  ng_sigma = ceil(Int64, (sigma_ub - sigma_lb)/0.1) + 1
  grid_sigma = Array(range(sigma_lb, sigma_ub, length = ng_sigma))
  GaussBlur2D(sigma_lb, sigma_ub, np, 2np, psf_thresh, grid_f, grid_sigma, 2)
end

function getStartingPoint(model :: GaussBlur2D, r_vec :: Vector{Float64})
  r = reshape(r_vec, model.n_pixels, model.n_pixels)
  ng = model.ng
  grid_objective_values = model.grid_f'*r*model.grid_f
  best_point_idx = argmax(grid_objective_values)
  thetas =  [best_point_idx[2];best_point_idx[1]].*(model.n_pixels/model.ng)

  #grid sigma, fit weight with least squares, choose best combo
  opt_w = 0
  opt_s = model.sigma_lb
  opt_obj = Inf
  for s in range(model.sigma_lb, model.sigma_ub, length=20)
    coords_sigma = reshape([thetas[1] thetas[2] s 1.0],4,1)
    A = vec(phi(model, coords_sigma))
    A = reshape(A, length(A), 1)
    l_fit = lm(A, r_vec)
    obj_v = sum(residuals(l_fit).^2)
    if obj_v < opt_obj
      opt_s = s
      opt_w = coef(l_fit)[1]
      opt_obj = obj_v
    end
  end
  if opt_w < 0
    opt_w = 0
  end

  push!(thetas, opt_s)
  push!(thetas, opt_w)
  return thetas
end



parameterBounds(model :: GaussBlur2D) = ([0.0,0.0,model.sigma_lb, 0.0],[Float64(model.n_pixels), Float64(model.n_pixels), model.sigma_ub,2^16-1.0])#[1.0,1.0])

"""
computeGradient(model :: GaussBlur2D, thetas :: Matrix{Float64}, v :: Vector{Float64})
    model - GaussBlur2D object
    thetas - matrix of coordinates and σ widths for the points. the first row contains x₁ coordinate along the vertical axis, second row
             contains the x₂ coordinate along the horizontal axis, and the third row contains the σ widths
    v - vector of the gradient of loss. Where r is the vector of resudiuals and ∇ = (∂/∂r₁ + ∂/∂r₂ + ... + ∂/∂rₙ)
     ∇(||r||₂) = r/||r||₂. rᵢ is the residual at pixel i.

     The goal of this function is to compute the gradient of the loss function with respect to the variable parameters. Here,
     ∇ = ∑ₖ(êₓ₁,ₖ ∂/∂x₁,ₖ +êₓ₂,ₖ ∂/∂x₂,ₖ+ê_σₖ ∂/∂σₖ), where x₁,ₖ is the coordinate along the vertical axis and x₂,ₖ is the coordinate along the horizontal axis
     of the kth dot. σₖ is the width parameter of the kth dot. Here we evaluate each of the partial derivatives. I will start with some definitions:

        yᵢⱼ, the obserced value of the pixel at i,j
        f(i,j,x₁ₖ,x₂ₖ,σₖ) = e^(-((i-x₁ₖ)² + (j-x₂ₖ)²)/(2σₖ²)), the value that the kth dot in the model contributes to pixel i,j
        f(i,x,σ) = e^(-((i-x)²/(2σ²)), the value that the kth dot in the model contributes to pixel i,j

        rᵢⱼ =  yᵢⱼ - ∑ₖ(wₖ f(i,x₁ₖ,σₖ)f(j,x₂ₖ,σₖ)),  the residual at pixel i,j
        R = {rᵢⱼ|i,j ∈ [1,n_pixels]}, the Matrix of residuals
        loss = ∑ᵢⱼ(rᵢⱼ²)
        ||r||₂ = √(∑ᵢⱼ(rᵢⱼ²)), the l-2 norm of the residuals
        ∂f/∂x(i,x,σ) = -(i-x)f(i,x,σ)/σ²
        ∂f/∂x₁ₖ(i,j,x₁ₖ,x₂ₖ,σₖ) = ∂f/∂x(i,x₁ₖ,σₖ)f(j,x₂ₖ,σₖ)
        ∂f/∂x₂ₖ(i,j,x₁ₖ,x₂ₖ,σₖ) = f(i,x₁ₖ,σₖ)∂f/∂x(j,x₂ₖ,σₖ)
        ∂loss/∂x₁ₖ = 2 ∑ᵢⱼ rᵢⱼ wₖ ∂f/∂x₁ₖ(i,j,x₁ₖ,x₂ₖ,σₖ)
        ∂loss/∂x₂ₖ = 2 ∑ᵢⱼ rᵢⱼ wₖ ∂f/∂x₂ₖ(i,j,x₁ₖ,x₂ₖ,σₖ)
        ∂loss/∂σₖ = -∑ᵢⱼ((i-x₁ₖ)rᵢⱼ wₖ ∂f/∂x₁ₖ(i,j,x₁ₖ,x₂ₖ,σₖ) + (j-x₂ₖ) rᵢⱼ wₖ ∂f/∂x₂ₖ(i,j,x₁ₖ,x₂ₖ,σₖ))/σₖ


"""
function computeGradient(model :: GaussBlur2D, thetas :: Matrix{Float64}, r :: Vector{Float64})

  # v is the gradient of loss (vi = resid_i/norm(residuals))
  R = reshape(r, model.n_pixels, model.n_pixels)

  gradient = zeros(size(thetas))
  #allocate temporary variables...
  f_x1 = zeros(model.n_pixels)
  f_x2 = zeros(model.n_pixels)
  fpx2 = zeros(model.n_pixels)
  fpx1 = zeros(model.n_pixels)
  fpx1s = zeros(model.n_pixels)
  fpx2s = zeros(model.n_pixels)
  v_x1 = zeros(model.n_pixels)
  v_x2 = zeros(model.n_pixels)

  #compute gradient
  for k = 1:size(thetas,2)
    point = vec(thetas[:,k])
    x₁ₖ, x₂ₖ, σₖ, wₖ = point
    computeFG!(model, x₁ₖ, σₖ, f_x1, fpx1, fpx1s)
    computeFG!(model, x₂ₖ, σₖ, f_x2, fpx2, fpx2s)
    ∂loss∂x₁ₖ = - 2 * wₖ * (f_x2' * R * fpx1)/(σₖ^2)
    ∂loss∂x₂ₖ = - 2 * wₖ * (fpx2' * R * f_x1)/(σₖ^2)
    ∂loss∂σₖ = - 2 * wₖ * ( f_x2' * R * fpx1s + fpx2s' * R * f_x1)/(σₖ^3)
    ∂loss∂wₖ = - 2 * (f_x2' * R * f_x1)
    gradient[:, k] = [∂loss∂x₁ₖ, ∂loss∂x₂ₖ, ∂loss∂σₖ, ∂loss∂wₖ]
  end
  return gradient
end

##not optimized.
function computeFs(x,n_pixels :: Int64, sigma, psf_thresh :: Int64, result :: Matrix{Float64}  = zeros(n_pixels,length(x)))
  nx = length(x)
  @simd for k = 1:nx
    sigma2_inv = 1.0/(2.0*sigma[k]^2)
    @inbounds start = maximum([1, round(Int64, x[k])-psf_thresh])
    @inbounds stop = minimum([n_pixels, round(Int64, x[k])+psf_thresh])
    @simd for i = start:stop
      @inbounds result[i, k] = exp(-sigma2_inv*(i-x[k])^2)
    end
  end
  return result
end


function computeFG!(s :: GaussBlur2D, x :: Float64, sigma :: Float64, f :: Vector{Float64}  = zeros(s.n_pixels), g :: Vector{Float64} = zeros(s.n_pixels), gs :: Vector{Float64} = zeros(s.n_pixels))
  n_pixels = s.n_pixels
  start = maximum([1, round(Int64, x)-s.psf_thresh])
  stop = minimum([n_pixels, round(Int64, x)+s.psf_thresh])
  sigma2_inv = 1.0/(2*sigma^2)

  #might wanna just bite the bullet and allocate here...
  @fastmath @inbounds @simd for i = start:stop
    f[i] = exp(-sigma2_inv*(i-x)^2)
    g[i] = (i-x)*f[i]
    gs[i] = g[i]*(i-x)
  end
end

function phi(s :: GaussBlur2D, parameters :: Matrix{Float64})
  n_pixels = s.n_pixels
  if size(parameters,2) == 0
    return zeros(n_pixels*n_pixels)
  end
  v_x = computeFs(vec(parameters[1,:]),n_pixels,parameters[3,:], s.psf_thresh)
  v_y = computeFs(vec(parameters[2,:]),n_pixels,parameters[3,:], s.psf_thresh)
  v_x .= v_x * diagm(parameters[4,:])

  return vec(v_y*v_x')
end

function lmo(model :: GaussBlur2D, r :: Vector{Float64})
  lb_0,ub_0 = parameterBounds(model)
  initial_x = getStartingPoint(model, r)
  lb1 = maximum([lb_0[1], initial_x[1] - 0.5])
  lb2 = maximum([lb_0[2], initial_x[2] - 0.5])
  ub1 = minimum([ub_0[1], initial_x[1] + 0.5])
  ub2 = minimum([ub_0[2], initial_x[2] + 0.5])
  lb = [lb1, lb2, initial_x[3], initial_x[4]]
  ub = [ub1, ub2, initial_x[3], initial_x[4]]
  p = length(lb)

  optx, optf = localDescent(model, LSLoss(), reshape(initial_x, 4, 1), [1.0], r, (lb, ub))

  lb = [optx[1], optx[2], lb_0[3], 0.0]
  ub = [optx[1], optx[2], ub_0[3], initial_x[4]*10]
  optx, optf = localDescent(model, LSLoss(), optx, [1.0], r, (lb, ub))
 
  return (optx,optf)
end

function localDescent_coord(s :: GaussBlur2D, lossFn :: Loss, thetas ::Matrix{Float64}, w :: Vector{Float64}, y :: Vector{Float64})
  lb,ub = parameterBounds(s)
  nPoints = size(thetas,2)
  im_lb = fill(lb[1], nPoints)
  im_ub = fill(ub[2], nPoints)
  x1_lb = broadcast((a, b) -> a > b ? a : b, im_lb, thetas[1,:] .- s.psf_thresh/2)
  x2_lb = broadcast((a, b) -> a > b ? a : b, im_lb, thetas[2,:] .- s.psf_thresh/2)
  x1_ub = broadcast((a, b) -> a < b ? a : b, im_ub, thetas[1,:] .+ s.psf_thresh/2)
  x2_ub = broadcast((a, b) -> a < b ? a : b, im_ub, thetas[2,:] .+ s.psf_thresh/2)

  lb = vec(vcat(x1_lb', x2_lb'))
  ub = vec(vcat(x1_ub', x2_ub'))

  p = size(thetas,1)
  su = SupportUpdateProblem(nPoints,p,s,y,w,lossFn)
  function f_and_g!(x,g)
      ps = reshape(x, 2, Int64(length(x)/2))
      ps = vcat(ps, thetas[3:4,:])
      points = reshape(ps,su.p,su.nPoints)
      output = phi(s,points)
      residual = su.y .- output
      l = sum(residual.^2)
      g[:] = computeGradient(s, points, residual)[1:2, :]
      l
  end
  opt = Opt(NLopt.LD_MMA, 2*nPoints)
  initializeOptimizer!(s, opt)
  min_objective!(opt, f_and_g!)
  lower_bounds!(opt, lb)
  upper_bounds!(opt, ub)
  (optf,optx,ret) = optimize(opt, vec(thetas[1:2,:]))
  if ret == :FORCED_STOP
    error("Forced Stop in Coordinate Optimization")
  end
  return reshape(optx,2,nPoints), optf
end

function localDescent_sigma(s :: GaussBlur2D, lossFn :: Loss, thetas ::Matrix{Float64}, w :: Vector{Float64}, y :: Vector{Float64})
  lb,ub = parameterBounds(s)
  s_lb = lb[3]
  s_ub = ub[3]
  w_lb = 0
  w_ub = thetas[4,:].*5


  nPoints = size(thetas,2)

  lb = repeat([s_lb, w_lb], nPoints)
  ub = vec(vcat(fill(s_ub, nPoints)', w_ub'))
  p = size(thetas,1)
  su = SupportUpdateProblem(nPoints,p,s,y,w,lossFn)

  #coordinates are fixed, sigma is input parameter to be optimized
  function f_and_g!(sigma_weight,g)
      crds = thetas[1:2,:]
      sws = reshape(sigma_weight, 2, Int64(length(sigma_weight)/2))
      ps = vcat(crds, sws)

      points = reshape(ps,su.p,su.nPoints)
      output = phi(su.s,points)
      residual = su.y .- output
      l = sum(residual.^2)
      g[:] = computeGradient(s, points, residual)[3:4, :]
      l
  end
  opt = Opt(NLopt.LD_MMA, 2*nPoints)
  initializeOptimizer!(s, opt)
  min_objective!(opt, f_and_g!)
  lower_bounds!(opt, lb)
  upper_bounds!(opt, ub)
  (optf,optx,ret) = optimize(opt, vec(thetas[3:4, :]))
  if ret == :FORCED_STOP
    error("Forced Stop in sigma weight Optimization")
  end

  return reshape(optx,2, nPoints), optf
end

make_KDTree(model :: GaussBlur2D, pnts :: DataFrame) = KDTree(Array([pnts.x pnts.y]'))


function initialize_dot_records(model :: GaussBlur, initial_ps :: Matrix)
  initial_ps 
  if length(initial_ps) > 0
    tree = KDTree(initial_ps[1:2,:])
    initial_ps_df = DataFrame(initial_ps', [:x, :y, :s, :w])
    records = copy(initial_ps_df)
    records.highest_mw = [initial_ps_df.w]
    records.lowest_mw = [initial_ps_df.w]
    last_iter = copy(initial_ps_df)
    last_iter.records_idxs = [1]
  else 
    records = DataFrame(x=[],y=[],s=[],w=[],highest_mw=[],lowest_mw=[])
    last_iter = DataFrame(x=[],y=[],s=[],w=[], records_idxs=[])
    tree = KDTree([Inf; Inf])
  end  
  return DotRecords(records, last_iter, tree)
end

function update_records!(model :: GaussBlur, records :: DotRecords, new_iteration :: Matrix, ϵ :: Float64, dims)
  #@bp
  # search for matches from last iteration
  #idxs, dists = knn(records.last_iteration_tree, new_iteration[1:2, :], 1)
  idxs, dists = knn(records.last_iteration_tree, new_iteration[1:dims, :], 1)

  dists = getindex.(dists,1)
  idxs = getindex.(idxs,1)

  nrows, ncols = size(new_iteration)
  idxs_new = Array(1:ncols)
  #@bp
  matched_idxs = idxs_new[dists .<= ϵ]
  new_dot_idxs = idxs_new[dists .> ϵ]

  old_matched_idxs = idxs[dists .<= ϵ]
  old_unmatched_idxs = filter(i -> i ∉ old_matched_idxs, Array(1:nrow(records.last_iteration)))
  unmatched_records = records.last_iteration.records_idxs[old_unmatched_idxs]

  # update records
  mw = minimum(new_iteration[nrows,:])
  if dims == 2
    new_iteration = length(new_iteration) > 0 ? DataFrame(new_iteration', [:x, :y, :s, :w]) : DataFrame(x=[],y=[],s=[],w=[])
  elseif dims == 3
    new_iteration = length(new_iteration) > 0 ? DataFrame(new_iteration', [:x, :y, :z, :sxy, :sz, :w]) : DataFrame(x=[],y=[],z=[],sxy=[],sz=[],w=[])
  else
    error("dims = $dims. Must be either 2 or 3.")
  end
  records.records[unmatched_records, "lowest_mw"] .= mw
  
  if length(matched_idxs) > 0
    matched_dot_record_idxs = records.last_iteration.records_idxs[old_matched_idxs]
  end
  
  new_dots = new_iteration[new_dot_idxs,:]
  new_dots[!, "lowest_mw"] .= 0
  new_dots[!, "highest_mw"] .= mw
  records.records = vcat(records.records, new_dots)

  #update last iteration
  new_iteration[!, "records_idxs"] .= 0
  if length(matched_idxs) > 0    
    new_iteration[matched_idxs, "records_idxs"] .= matched_dot_record_idxs
  end
  new_iteration[new_dot_idxs, "records_idxs"] .= Array((nrow(records.records)-nrow(new_dots)+1):nrow(records.records))
  records.last_iteration = new_iteration
  records.last_iteration_tree = make_KDTree(model, records.last_iteration)

  return records
end

function get_close_network(model :: GaussBlur2D, thetas)
  pnts_mat = Array(thetas[1:2,:])
  ndims, final_pnt_ind = size(pnts_mat)

  if final_pnt_ind == 1
    return [1], []
  elseif final_pnt_ind == 2
    if sqrt(sum((pnts_mat[:,1] - pnts_mat[:,2]).^2)) < model.psf_thresh
      return [1,2], []
    else
      return [2], [1]
    end
  else
    dbr = dbscan(pnts_mat, model.psf_thresh, min_neighbors=1, min_cluster_size=1)
    dbscan_clusters = [sort(vcat(dbc.core_indices,  dbc.boundary_indices)) for dbc in dbr]
    for cluster in dbscan_clusters
      if final_pnt_ind ∈ cluster
        not_cluster = filter(i -> i ∉ cluster, Array(1:final_pnt_ind))
        return cluster, not_cluster
      end
    end
  end
end