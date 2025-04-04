export GaussBlur3D


struct GaussBlur3D <: GaussBlur
  sigma_z_lb :: Float64
  sigma_z_ub :: Float64
  n_pixels :: Int64
  n_slices :: Int64
  psf_z_thresh :: Int64
  zgrid :: Matrix{Float64}
  gb2d :: GaussBlur2D
  dims :: Int64
end

function GaussBlur3D(
                     sigma_xy_lb :: Float64,
                     sigma_xy_ub :: Float64,
                     sigma_z_lb :: Float64,
                     sigma_z_ub :: Float64,
                     n_pixels :: Int64,
                     n_slices :: Int64
                     )
  gb2d = GaussBlur2D(sigma_xy_lb, sigma_xy_ub, n_pixels)
  psf_z_thresh = ceil(Int64, sigma_z_ub*3.0)
  zgrid = computeFs(Array(0.5:0.5:n_slices), n_slices, ones(2*n_slices).*sigma_z_lb, psf_z_thresh)
  GaussBlur3D(sigma_z_lb, sigma_z_ub, n_pixels, n_slices, psf_z_thresh, zgrid, gb2d, 3)
end


function getStartingPoint(model :: GaussBlur3D, r_vec :: Vector{Float64})
  r = reshape(r_vec, model.n_pixels, model.n_pixels, model.n_slices)
  #ng = model.ng
  z_slice_obj_values = zeros(model.n_pixels*2, model.n_pixels*2, model.n_slices)
  for z = 1:(model.n_slices)
    z_slice_obj_values[:,:,z] = model.gb2d.grid_f'*r[:,:,z]*model.gb2d.grid_f
  end

  grid_objective_values = zeros(model.n_pixels*2, model.n_pixels*2, 2*model.n_slices)
  for y = 1:(2*model.n_pixels)# n_z_slices
    grid_objective_values[y,:,:] = z_slice_obj_values[y,:,:]*model.zgrid
  end

  best_point_idx = argmax(grid_objective_values) #argmin(grid_objective_values)
  thetas =  [best_point_idx[2], best_point_idx[1]].*(model.n_pixels/model.gb2d.ng)
  push!(thetas, best_point_idx[3]/2)
  #ToDo: grid sigma
  """
  function grid_sigma(trial_sigma)
      t_thetas = reshape(cat(thetas, [trial_sigma], dims = 1),3,1)
      model_sigma = phi(model, t_thetas, [1.0])
      sum((model_sigma .- v_vec).^2)
  end
  sse = map(grid_sigma, model.grid_sigma)
  println("sse: ", sse)
  sigma_start = model.grid_sigma[argmin(sse)]
  """
  opt_w = 0
  opt_s_xy = model.gb2d.sigma_lb
  opt_s_z = model.sigma_z_lb
  opt_obj = Inf
  for s_xy in range(model.gb2d.sigma_lb, model.gb2d.sigma_ub, length=20), s_z in range(model.sigma_z_lb, model.sigma_z_ub, length=10)
    coords_sigma = reshape([thetas[1] thetas[2] thetas[3] s_xy s_z 1.0],6,1)
    A = vec(phi(model, coords_sigma))
    A = reshape(A, length(A), 1)
    l_fit = lm(A, r_vec)
    obj_v = sum(residuals(l_fit).^2)
    if obj_v < opt_obj
      opt_s_xy = s_xy
      opt_s_z = s_z
      opt_w = coef(l_fit)[1]
      opt_obj = obj_v
    end
  end
  if opt_w < 0
    opt_w = 0
  end
  push!(thetas, opt_s_xy)
  push!(thetas, opt_s_z)
  push!(thetas, opt_w)
  return thetas
end


function getNextPoints(model :: GaussBlur3D, r_vec :: Vector{Float64}, min_weight)
  r = reshape(r_vec, model.n_pixels, model.n_pixels, model.n_slices)
  #ng = model.ng
  z_slice_obj_values = zeros(model.n_pixels*2, model.n_pixels*2, model.n_slices)
  for z = 1:(model.n_slices)
    z_slice_obj_values[:,:,z] = model.gb2d.grid_f'*r[:,:,z]*model.gb2d.grid_f
  end

  grid_objective_values = zeros(model.n_pixels*2, model.n_pixels*2, 2*model.n_slices)
  for y = 1:(2*model.n_pixels)# n_z_slices
    grid_objective_values[y,:,:] = z_slice_obj_values[y,:,:]*model.zgrid
  end

  local_maxima_mask = local_maxima(grid_objective_values) #argmin(grid_objective_values)
  next_pnt_coords = findall(local_maxima_mask .!= 0)
  next_pnts= hcat(collect.(Tuple.(next_pnt_coords))...)
  thetas = hcat(next_pnts[2, :], next_pnts[1, :])' .*(model.n_pixels/model.gb2d.ng)
  #thetas =  [next_pnts[2], next_pnts[1]].*(model.n_pixels/model.gb2d.ng)
  thetas = vcat(thetas, next_pnts[3, :]' ./ 2)
  
  ndots = length(next_pnt_coords)

  thetas = vcat(thetas, fill(model.gb2d.sigma_lb, 1, ndots))
  thetas = vcat(thetas, fill(model.sigma_z_lb, 1, ndots))
  thetas = vcat(thetas, ones(1, ndots))

  opt_w = 0
  opt_s_xy = model.gb2d.sigma_lb
  opt_s_z = model.sigma_z_lb
  opt_obj = Inf
  psfs = []
  for i in 1:ndots
    coords_sigma = reshape(thetas[:,i], 6,1) #reshape([thetas[1] thetas[2] model.sigma_lb 1.0],4,1)
    A = vec(phi(model, coords_sigma))
    A = reshape(A, length(A), 1)
    push!(psfs, A)
  end
  #=
  for s_xy in range(model.gb2d.sigma_lb, model.gb2d.sigma_ub, length=20), s_z in range(model.sigma_z_lb, model.sigma_z_ub, length=10)
    coords_sigma = reshape([thetas[1] thetas[2] thetas[3] s_xy s_z 1.0],6,1)
    A = vec(phi(model, coords_sigma))
    A = reshape(A, length(A), 1)
    l_fit = lm(A, r_vec)
    obj_v = sum(residuals(l_fit).^2)
    if obj_v < opt_obj
      opt_s_xy = s_xy
      opt_s_z = s_z
      opt_w = coef(l_fit)[1]
      opt_obj = obj_v
    end
  end=#
  X = hcat(psfs...)
  l_fit = lm(X, r_vec)
  thetas[6,:] .= coef(l_fit)
  thetas = thetas[:, coef(l_fit) .> min_weight]

  #=
  if opt_w < 0
    opt_w = 0
  end
  push!(thetas, opt_s_xy)
  push!(thetas, opt_s_z)
  push!(thetas, opt_w)
  =#
  return thetas
end

function phi(s :: GaussBlur3D, parameters :: Matrix{Float64})
  canvas = zeros(s.n_pixels, s.n_pixels, s.n_slices)
  for i in 1:size(parameters)[2]
    dot_2d_params = reshape(parameters[[1,2,4, 6], i], 4,1)
    phi2d_i = phi(s.gb2d, dot_2d_params)
    z_profile = computeFs([parameters[3,i]], s.n_slices, [parameters[5,i]], s.psf_z_thresh)
    phi2d_i = reshape(phi2d_i, s.n_pixels, s.n_pixels, 1)
    z_profile = reshape(z_profile, 1, 1, s.n_slices)
    dot_psf = phi2d_i .* z_profile
    canvas .+= dot_psf
  end
  return vec(canvas)
end

function parameterBounds(m :: GaussBlur3D)
  lbs = [0.0,0.0,0.0,m.gb2d.sigma_lb,m.sigma_z_lb, 0]
  ubs = [Float64(m.n_pixels), Float64(m.n_pixels), Float64(m.n_slices),m.gb2d.sigma_ub,m.sigma_z_ub, Inf]
  (lbs, ubs)
end

#function computeGradient(model :: GaussBlur3D, weights :: Vector{Float64}, thetas :: Matrix{Float64}, r :: Vector{Float64})
function computeGradient(model :: GaussBlur3D, thetas :: Matrix{Float64}, r :: Vector{Float64})
  # r is the gradient of loss (vi = resid_i/norm(residuals))
  r = reshape(r, model.n_pixels, model.n_pixels, model.n_slices)

  gradient = zeros(size(thetas))

  #allocate temporary variables...
  f_x1 = zeros(model.n_pixels)
  f_x2 = zeros(model.n_pixels)
  fpx2 = zeros(model.n_pixels)
  fpx1 = zeros(model.n_pixels)
  fpx1s = zeros(model.n_pixels)
  fpx2s = zeros(model.n_pixels)

  #compute gradient
  for l = 1:size(thetas,2)
    point = vec(thetas[:,l])
    x₁ₗ, x₂ₗ, x₃ₗ, σₓₗ, σzₗ, wₗ = vec(thetas[:, l])
    vec(thetas[:,l])
    computeFG!(model.gb2d, point[1], point[4], f_x1, fpx1, fpx1s)
    computeFG!(model.gb2d, point[2], point[4], f_x2, fpx2, fpx2s)



    k_sums = zeros(size(thetas)[1])
    for k = 1:model.n_slices
      fz = exp(-(k-x₃ₗ)^2/(2 * σzₗ^2))

      ∂loss∂x₁ₗ = - 2 * wₗ * fz *(f_x2' * r[:,:,k] * fpx1)/(σₓₗ^2)
      ∂loss∂x₂ₗ = - 2 * wₗ * fz * (fpx2' * r[:,:,k] * f_x1)/(σₓₗ^2)
      ∂loss∂σₓₗ = - 2* wₗ * fz * ( f_x2' * r[:,:,k] * fpx1s + fpx2s' * r[:,:,k] * f_x1)/(σₓₗ^3)
      ∂loss∂x₃ₗ = - 2 * wₗ * (k - x₃ₗ) * fz * (f_x2' * r[:,:,k] * f_x1)/(σzₗ^2)
      ∂loss∂σzₗ = - 2 * wₗ * (k - x₃ₗ)^2 * fz * (f_x2' * r[:,:,k] * f_x1)/(σzₗ^3)
      ∂loss∂wₗ = - 2 * fz* (f_x2' * r[:,:,k] * f_x1)

      k_sums .+= [∂loss∂x₁ₗ, ∂loss∂x₂ₗ, ∂loss∂x₃ₗ, ∂loss∂σₓₗ, ∂loss∂σzₗ, ∂loss∂wₗ]
    end
    gradient[:, l] = k_sums
  end
  return gradient
end


function localDescent_coord(s :: GaussBlur3D, lossFn :: Loss, thetas ::Matrix{Float64}, w :: Vector{Float64}, y :: Vector{Float64})
  lb,ub = parameterBounds(s)
  nPoints = size(thetas,2)
  im_lb_1 = fill(lb[1], nPoints)
  im_lb_2 = fill(lb[2], nPoints)
  im_lb_3 = fill(lb[3], nPoints)
  im_ub_1 = fill(ub[1], nPoints)
  im_ub_2 = fill(ub[2], nPoints)
  im_ub_3 = fill(ub[3], nPoints)

  thetas[1,:]

  x1_lb = broadcast((a, b) -> a > b ? a : b, im_lb_1, thetas[1,:] .- s.gb2d.psf_thresh)
  x2_lb = broadcast((a, b) -> a > b ? a : b, im_lb_2, thetas[2,:] .- s.gb2d.psf_thresh)
  x3_lb = broadcast((a, b) -> a > b ? a : b, im_lb_3, thetas[3,:] .- s.psf_z_thresh)
  x1_ub = broadcast((a, b) -> a < b ? a : b, im_ub_1, thetas[1,:] .+ s.gb2d.psf_thresh)
  x2_ub = broadcast((a, b) -> a < b ? a : b, im_ub_2, thetas[2,:] .+ s.gb2d.psf_thresh)
  x3_ub = broadcast((a, b) -> a < b ? a : b, im_ub_3, thetas[3,:] .+ s.psf_z_thresh)

  lb = vec(vcat(x1_lb', x2_lb', x3_lb'))
  ub = vec(vcat(x1_ub', x2_ub', x3_ub'))

  thetas_copy = copy(thetas)

  p = size(thetas,1)
  su = SupportUpdateProblem(nPoints,p,s,y,w,lossFn)
  function f_and_g!(x,g)
      ps = reshape(x, 3, Int64(length(x)/3))
      thetas_copy[1:3, :] .= ps
      #output = phi(su.s,ps)
      output = phi(su.s,thetas_copy)
      residual = su.y .- output
      residual = su.y .- output
      #l,v_star = loss(s.lossFn,residual)
      #g[:] = computeGradient(su.s, ps, residual)[1:3]
      l,v_star = loss(su.lossFn,residual)
      g[:] = vec(computeGradient(su.s, thetas_copy, residual)[1:3, :])
      return l
  end
  opt = Opt(NLopt.LD_MMA, 3*nPoints)
  initializeOptimizer!(s, opt)
  min_objective!(opt, f_and_g!)
  lower_bounds!(opt, lb)
  upper_bounds!(opt, ub)
  (optf,optx,ret) = optimize(opt, vec(thetas[1:3,:]))
  return reshape(optx,3,nPoints), optf
end

function localDescent_sigma(s :: GaussBlur3D, lossFn :: Loss, thetas ::Matrix{Float64}, w :: Vector{Float64}, y :: Vector{Float64})
  lb = [s.gb2d.sigma_lb, s.sigma_z_lb, 0]
  ub = [s.gb2d.sigma_ub, s.sigma_z_ub, Inf]

  nPoints = size(thetas,2)
  p = size(thetas,1)
  su = SupportUpdateProblem(nPoints,p,s,y,w,lossFn)

  #coordinates are fixed, sigma is input parameter to be optimized
  function f_and_g!(sigmas_weights,g)
      sigmas_weights = reshape(sigmas_weights, 3, nPoints)
      ps = vcat(thetas[1:3,:], sigmas_weights)
      output = phi(su.s,ps)
      residual = su.y .- output
      l,v_star = loss(su.lossFn,residual)
      sw_g = computeGradient(su.s, ps, residual)[4:6, :]
      g[:] = sw_g 
      return l
  end
  opt = Opt(NLopt.LD_MMA, 3*nPoints)
  initializeOptimizer!(s, opt)
  min_objective!(opt, f_and_g!)
  lower_bounds!(opt, vec(repeat(lb, 1, nPoints)))
  upper_bounds!(opt, vec(repeat(ub, 1, nPoints)))
  (optf,optx,ret) = optimize(opt, vec(thetas[4:6, :]))
  if ret == :FORCED_STOP
    error("Forced Stop in sigma weight Optimization")
  end
  return reshape(optx, 3, nPoints), optf
end


function localDescent(s :: GaussBlur3D, lossFn :: Loss, thetas ::Matrix{Float64}, y :: Vector{Float64}, bounds = parameterBounds(s))
  lb, ub = bounds
  nPoints = size(thetas,2)

  p = size(thetas,1)
  w = thetas[:,end]
  su = SupportUpdateProblem(nPoints,p,s,y,w,lossFn)
  f_and_g!(x,g) = localDescent_f_and_g!(x,g,su)
  opt = Opt(NLopt.LD_MMA, length(thetas))
  initializeOptimizer!(s, opt)
  min_objective!(opt, f_and_g!)
  lower_bounds!(opt, vec(repeat(lb,1,nPoints)))
  upper_bounds!(opt, vec(repeat(ub,1,nPoints)))
  (optf,optx,ret) = optimize(opt, vec(thetas))
  return (reshape(optx,p,nPoints), optf, ret)
end


function lmo(model :: GaussBlur3D, r :: Vector{Float64})
  lb_0,ub_0 = parameterBounds(model)
  initial_x = getStartingPoint(model, r)
  lb1 = maximum([lb_0[1], initial_x[1] - 0.5])
  lb2 = maximum([lb_0[2], initial_x[2] - 0.5])
  lb3 = maximum([lb_0[3], initial_x[3] - 0.5])

  ub1 = minimum([ub_0[1], initial_x[1] + 0.5])
  ub2 = minimum([ub_0[2], initial_x[2] + 0.5])
  ub3 = minimum([ub_0[3], initial_x[3] + 0.5])

  lb = [lb1, lb2, lb3, initial_x[4], initial_x[5], initial_x[6]]
  ub = [ub1, ub2, ub3, initial_x[4], initial_x[5], initial_x[6]]
  optx, optf, ret = localDescent(model, LSLoss(), reshape(initial_x, 6, 1), r, [lb, ub])

  if ret == :FORCED_STOP
    error("Forced Stop in Coordinate Optimization")
  end

  lb = [optx[1], optx[2], optx[3], lb_0[4], lb_0[5], lb_0[6]]
  ub = [optx[1], optx[2], optx[3], ub_0[4], ub_0[5], ub_0[6]]

  optx, optf, ret = localDescent(model, LSLoss(), optx, r, [lb, ub])
  if ret == :FORCED_STOP
    error("Forced Stop in Coordinate Optimization")
  end
  return (optx, optf)
end

make_KDTree(model :: GaussBlur3D, pnts :: DataFrame) = KDTree(Array([pnts.x pnts.y pnts.z]'))


function initialize_dot_records(model :: GaussBlur3D, initial_ps :: Matrix)
  initial_ps 
  if length(initial_ps) > 0
    tree = KDTree(initial_ps[1:2,:])
    initial_ps_df = DataFrame(initial_ps', [:x, :y, :z, :sxy, :sz, :w])
    records = copy(initial_ps_df)
    records.highest_mw = [initial_ps_df.w]
    records.lowest_mw = [initial_ps_df.w]
    last_iter = copy(initial_ps_df)
    last_iter.records_idxs = [1]
  else 
    records = DataFrame(x=[],y=[],z=[],sxy=[],sz=[],w=[],highest_mw=[],lowest_mw=[])
    last_iter = DataFrame(x=[],y=[],z=[],sxy=[],sz=[],w=[],records_idxs=[])
    tree = KDTree([Inf; Inf; Inf])
  end  
  return DotRecords(records, last_iter, tree)
end
