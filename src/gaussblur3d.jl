export GaussBlur3D
using JumP
using GLPK

struct GaussBlur3D <: GaussBlur
  sigma_z_lb :: Float64
  sigma_z_ub :: Float64
  n_pixels :: Int64
  n_slices :: Int64
  psf_z_thresh :: Int64
  zgrid :: Matrix{Float64}
  gb2d :: GaussBlur2D
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
  GaussBlur3D(sigma_z_lb, sigma_z_ub, n_pixels, n_slices, psf_z_thresh, zgrid, gb2d)
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
  fx1 = zeros(model.n_pixels)
  fx2 = zeros(model.n_pixels)
  #fpx2 = zeros(model.n_pixels)
  ∂f_∂x₁ = zeros(model.n_pixels)
  ∂f_∂x₂= zeros(model.n_pixels)
  #fpx1 = zeros(model.n_pixels)
  #fpx1s = zeros(model.n_pixels)
  ∂²f_∂x₁² = zeros(model.n_pixels)
  #fpx2s = zeros(model.n_pixels)
  ∂²f_∂x₂² = zeros(model.n_pixels)

  #compute gradient
  for l = 1:size(thetas,2)
    x₁, x₂, x₃, σₓ₁₂, σₓ₃, w = vec(thetas[:,l])
    #point = vec(thetas[:,l])
    #x₁, x₂, x₃, σₓ₁₂, σₓ₃, w = point
    computeFG!(model.gb2d, x₁, σₓ₁₂, fx1, ∂f_∂x₁, ∂²f_∂x₁²)
    computeFG!(model.gb2d, x₂, σₓ₁₂, fx2, ∂f_∂x₂, ∂²f_∂x₂²)



    k_sums = zeros(size(thetas)[1])
    for k = 1:model.n_slices
      fz = exp(-(k-x₃)^2/(2 * σₓ₃^2))

      ∂loss∂x₁ₗ = - w * fz *(fx2' * r[:,:,k] * ∂f_∂x₁)/(σₓ₁₂^2)
      ∂loss∂x₂ₗ = - w * fz * (∂f_∂x₂' * r[:,:,k] * fx1)/(σₓ₁₂^2)
      ∂loss∂σₓₗ = - w * fz * ( fx2' * r[:,:,k] * ∂²f_∂x₁² + ∂²f_∂x₂²' * r[:,:,k] * fx1)/(σₓ₁₂^3)
      ∂loss∂x₃ₗ = - w * (k - x₃) * fz * (fx2' * r[:,:,k] * fx1)/(σₓ₃^2)
      ∂loss∂σzₗ = - w * (k - x₃)^2 * fz* (fx2' * r[:,:,k] * fx1)/(x₃^3)
      ∂loss∂wₖ = - fz* (fx2' * r[:,:,k] * fx1)

      k_sums .+= [∂loss∂x₁ₗ, ∂loss∂x₂ₗ, ∂loss∂x₃ₗ, ∂loss∂σₓₗ, ∂loss∂σzₗ, ∂loss∂wₖ]
    end
    #gradient[:, l] = [∂loss∂x₁ₗ, ∂loss∂x₂ₗ, ∂loss∂σₗ]
    gradient[:, l] = k_sums ./ model.n_slices
  end
  return gradient
end

get_greater(a, b) = maximum([a,b])
get_lesser(a, b) = minimum([a,b])

function localDescent_coord(s :: GaussBlur3D, lossFn :: Loss, thetas ::Matrix{Float64}, w :: Vector{Float64}, y :: Vector{Float64})
  lb,ub = parameterBounds(s)
  nPoints = size(thetas,2)
  im_lb = fill(lb[1], nPoints)
  im_ub = fill(ub[2], nPoints)
  #x1_lb = broadcast((a, b) -> a > b ? a : b, im_lb, thetas[1,:] .- s.gb2d.psf_thresh)
  #x2_lb = broadcast((a, b) -> a > b ? a : b, im_lb, thetas[2,:] .- s.gb2d.psf_thresh)
  #x3_lb = broadcast((a, b) -> a > b ? a : b, im_lb, thetas[3,:] .- s.psf_z_thresh)
  #x1_ub = broadcast((a, b) -> a < b ? a : b, im_ub, thetas[1,:] .+ s.gb2d.psf_thresh)
  #x2_ub = broadcast((a, b) -> a < b ? a : b, im_ub, thetas[2,:] .+ s.gb2d.psf_thresh)
  #x3_ub = broadcast((a, b) -> a < b ? a : b, im_ub, thetas[3,:] .+ s.psf_z_thresh)

  x1_lb = get_greater.(im_lb, thetas[1,:] .- s.gb2d.psf_thresh)
  x2_lb = get_greater.(im_lb, thetas[2,:] .- s.gb2d.psf_thresh)
  x3_lb = get_greater.(im_lb, thetas[3,:] .- s.psf_z_thresh)
  x1_ub = get_lesser.(im_ub, thetas[1,:] .+ s.gb2d.psf_thresh)
  x2_ub = get_lesser.(im_ub, thetas[2,:] .+ s.gb2d.psf_thresh)
  x3_ub = get_lesser.(im_ub, thetas[3,:] .+ s.psf_z_thresh)

  model = Model(GLPK.Optimizer)

  @variable(model, x1=[1:nPoints], start=thetas[1,:])
  @variable(model, x2=[1:nPoints], start=thetas[2,:])
  @variable(model, x3=[1:nPoints], start=thetas[3,:])
  σxy = theta[4,:] 
  σz = theta[5,:]
  w = theta[6,:]

  @constraint(model, x1_lb .<= x1)
  @constraint(model, x2_lb .<= x2)
  @constraint(model, x3_lb .<= x3)
  @constraint(model, x1 .<= x1_ub)
  @constraint(model, x2 .<= x2_ub)
  @constraint(model, x3 .<= x3_ub)

  @NLobjective(model, Min, sum([w[l]exp(-((x1[l]-i)^2 + (x2[l]-j)^2)/(2σxy[l]^2) -((x3[l]-k)^2)/(2σz[l]^2)) - y[i,j,k]]) for i in 

  #lb = vec(vcat(x1_lb', x2_lb', x3_lb'))
  #ub = vec(vcat(x1_ub', x2_ub', x3_ub'))

  p = size(thetas,1)
  su = SupportUpdateProblem(nPoints,p,s,y,w,lossFn)
  function f_and_g!(x,g)
      ps = reshape(x, 3, Int64(length(x)/3))
      output = phi(su.s,ps)
      residual = su.y .- output
      l,v_star = loss(s.lossFn,residual)
      g[:] = computeGradient(su.s, ps, residual)[1:3]
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