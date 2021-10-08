using SpecialFunctions
using LinearAlgebra
export GaussBlur2D
#using Lasso
using GLM

abstract type GaussBlur <: BoxConstrainedDifferentiableModel end

struct GaussBlur2D <: GaussBlur
  sigma_lb :: Float64
  sigma_ub :: Float64
  n_pixels :: Int64
  ng :: Int64 #this is small now
  psf_thresh :: Int64
  grid_f
  grid_sigma
end

function GaussBlur2D(sigma_lb, sigma_ub, np)
  psf_thresh = ceil(Int64, sigma_ub*3.0)
  #grid_f = computeFs(Array(0.5:0.5:np),np, ones(ng).*sigma_mid, psf_thresh)
  #
  grid_f = computeFs(Array(0.5:0.5:np),np, ones(2np).*sigma_lb, psf_thresh)
  ng_sigma = ceil(Int64, (sigma_ub - sigma_lb)/0.1) + 1
  grid_sigma = Array(range(sigma_lb, sigma_ub, length = ng_sigma))
  #GaussBlur2D(sigma_lb, sigma_ub, np, ng, psf_thresh, grid_f, grid_sigma)
  GaussBlur2D(sigma_lb, sigma_ub, np, 2np, psf_thresh, grid_f, grid_sigma)
end

function getStartingPoint(model :: GaussBlur2D, r_vec :: Vector{Float64})
  r = reshape(r_vec, model.n_pixels, model.n_pixels)
  ng = model.ng
  grid_objective_values = model.grid_f'*r*model.grid_f
  best_point_idx = argmax(grid_objective_values) #argmin(grid_objective_values)
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
  #push!(thetas, 1.0)
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
  #println("computing gradient...")
  #println("size(thetas): ", size(thetas))
  R = reshape(r, model.n_pixels, model.n_pixels)
  #Rt = R'

  #gradient = zeros(size(thetas)[1] + 1, size(thetas)[2])
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
    computeFG!(model, point[1], point[3], f_x1, fpx1, fpx1s)
    computeFG!(model, point[2], point[3], f_x2, fpx2, fpx2s)
    #∂loss∂x₁ₖ = - 2 * weights[k] * (f_x2' * R * fpx1)/(point[3]^2)
    #∂loss∂x₂ₖ = - 2 * weights[k] * (fpx2' * R * f_x1)/(point[3]^2)
    #∂loss∂σₖ = - 2 * weights[k] * ( f_x2' * R * fpx1s + fpx2s' * R * f_x1)/(point[3]^3)
    ∂loss∂x₁ₖ = - 2 * point[4] * (f_x2' * R * fpx1)/(point[3]^2)
    ∂loss∂x₂ₖ = - 2 * point[4] * (fpx2' * R * f_x1)/(point[3]^2)
    ∂loss∂σₖ = - 2 * point[4] * ( f_x2' * R * fpx1s + fpx2s' * R * f_x1)/(point[3]^3)
    ∂loss∂wₖ = - 2 * (f_x2' * R * f_x1)
    gradient[:, k] = [∂loss∂x₁ₖ, ∂loss∂x₂ₖ, ∂loss∂σₖ, ∂loss∂wₖ]
  end
  return gradient
end

##not optimized.
#function computeFs(x,n_pixels :: Int64, sigma :: Float64, psf_thresh :: Int64, result :: Matrix{Float64}  = zeros(n_pixels,length(x)))
function computeFs(x,n_pixels :: Int64, sigma, psf_thresh :: Int64, result :: Matrix{Float64}  = zeros(n_pixels,length(x)))
  nx = length(x)
  #sigma2_inv = 1.0/(2.0*sigma^2)
  @simd for k = 1:nx
    sigma2_inv = 1.0/(2.0*sigma[k]^2)
    @inbounds start = maximum([1, round(Int64, x[k])-psf_thresh])
    @inbounds stop = minimum([n_pixels, round(Int64, x[k])+psf_thresh])
    @simd for i = start:stop#1:n_pixels
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
  #prefactor_g = -1.0/(sigma^2)
  #prefactor_gs = 1.0/(sigma^3)

  #might wanna just bite the bullet and allocate here...
  #@fastmath @inbounds @simd  for i = 1:n_pixels
  @fastmath @inbounds @simd for i = start:stop
    f[i] = exp(-sigma2_inv*(i-x)^2)
    #g[i] = -prefactor_g*(i-x)*f[i]
    g[i] = (i-x)*f[i]
    gs[i] = g[i]*(i-x)
    #g_sig[i] = -g_spat[i]*(i-x)/sigma #prefactor_gs*f[i](i-x)^2
  end
end

function phi(s :: GaussBlur2D, parameters :: Matrix{Float64})
  n_pixels = s.n_pixels
  if size(parameters,2) == 0
    return zeros(n_pixels*n_pixels)
  end
  v_x = computeFs(vec(parameters[1,:]),n_pixels,parameters[3,:], s.psf_thresh)
  v_y = computeFs(vec(parameters[2,:]),n_pixels,parameters[3,:], s.psf_thresh)
  #scale!(v_x, weights)
  #v_x .= v_x * diagm(weights)
  v_x .= v_x * diagm(parameters[4,:])

  return vec(v_y*v_x')
end


function solveFiniteDimProblem(model :: GaussBlur, thetas, y, tau)#, fittype=LassoPath)
  nthetas = size(thetas)[2]
  nparams = size(thetas)[1]
  if nthetas == 0
    return Float64[]
  end

  #A = zeros(nthetas, model.n_pixels^2)
  if nparams == 3
    A = zeros(model.n_pixels^2, nthetas)
  elseif nparams == 5
    A = zeros(model.n_slices*model.n_pixels^2, nthetas)
  end

  for i = 1:nthetas
    #A[i, :] = vec(phi(model, reshape(thetas[:,i], 2, 1), [1.0]))
    A[:, i] = vec(phi(model, reshape(thetas[:,i], nparams, 1)))
  end

  #println("sum y: ", sum(y))
  #println("nthetas: ", nthetas)
  #results = fit(LassoPath, A, y)

  #results = fit(fittype, A, y)
  results = lm(A, y)
  return coef(results)
  """
  if fittype != LassoPath
    results = lm(A, y)
    println(results)
  else
    results = fit(fittype, A, y)
  end

  #results = fit(GeneralizedLinearModel, A, y)
  #println(results.coefs)
  n_lambdas = size(results.coefs)[2]
  coeffs = []
  for i = 1:(n_lambdas)
    sum_coeffs = sum(results.coefs[:, n_lambdas + 1 - i])
    #println("sum_coeffs ", i , ": ", sum_coeffs)
    if sum(results.coefs[:, n_lambdas + 1 - i]) <= tau
      println("lambda: ", i)
      println("sum(coefs): ", sum_coeffs)
      coeffs = Array(results.coefs[:, n_lambdas + 1 - i])
      break
    end
  end
  return coeffs
  #results = fit(LassoPath, A, y, λ = [5.0])
  #return Array(results.coefs[:,1])
  """
end

function lmo(model :: GaussBlur2D, r :: Vector{Float64})
  lb_0,ub_0 = parameterBounds(model)
  initial_x = getStartingPoint(model, r)
  lb1 = maximum([lb_0[1], initial_x[1] - 0.5])
  lb2 = maximum([lb_0[2], initial_x[2] - 0.5])
  ub1 = minimum([ub_0[1], initial_x[1] + 0.5])
  ub2 = minimum([ub_0[2], initial_x[2] + 0.5])
  lb = [lb1, lb2, initial_x[3], initial_x[4]]
  #ub = [ub1, ub2, ub[3]]
  ub = [ub1, ub2, initial_x[3], initial_x[4]]
  #println("Stating Point: ", initial_x)
  p = length(lb)
  #println("lb: ", lb)
  #println("ub: ", ub)
  optx, optf = localDescent(model, LSLoss(), reshape(initial_x, 4, 1), [1.0], r, (lb, ub))

  lb = [optx[1], optx[2], lb_0[3], 0.0]
  ub = [optx[1], optx[2], ub_0[3], initial_x[4]*10]
  optx, optf = localDescent(model, LSLoss(), optx, [1.0], r, (lb, ub))
  #println("lmo optx: ", optx)
"""
  function f_and_g!(point :: Vector{Float64}, gradient_storage :: Vector{Float64})
    #println("point: ", point)
    output = phi(model,reshape(point,p,1), [1.0])
    ip = dot(output,v)
    s = sign(ip)
    gradient_storage[:] = -s*computeGradient(model, [1.0],reshape(point,length(point),1), v)
    #gradient_storage[:] = -s*computeGradient(model, [1.0],reshape(point,length(point),1), ip)
    #println("gradient: ", gradient_storage)
    return -s*ip
  end


  function f_and_g!(points :: Vector{Float64}, gradient_storage :: Vector{Float64})
    println("f_and_g!")
    points = reshape(points,model.n_pixels,1)
    output = phi(model,points,[1.0])#s.w)
    residual = output .+ r
    l,v_star = loss(s.lossFn,residual)
    gradient_storage[:] = computeGradient(model.sim, [1.0], points, v_star)
    return l
  end

  opt = Opt(:LD_MMA, p)
  initializeOptimizer!(model, opt)
  min_objective!(opt, f_and_g!)
  lower_bounds!(opt, lb)
  upper_bounds!(opt, ub)
  (optf,optx,ret) = optimize(opt, initial_x)
  """
  #println("optx: ", optx)
  #println("ret: ", ret)
  return (optx,optf)
end

function localDescent_coord(s :: GaussBlur2D, lossFn :: Loss, thetas ::Matrix{Float64}, w :: Vector{Float64}, y :: Vector{Float64})
  lb,ub = parameterBounds(s)
  #lb, ub = bounds
  nPoints = size(thetas,2)
  im_lb = fill(lb[1], nPoints)
  im_ub = fill(ub[2], nPoints)
  x1_lb = broadcast((a, b) -> a > b ? a : b, im_lb, thetas[1,:] .- s.psf_thresh/2)
  x2_lb = broadcast((a, b) -> a > b ? a : b, im_lb, thetas[2,:] .- s.psf_thresh/2)
  x1_ub = broadcast((a, b) -> a < b ? a : b, im_ub, thetas[1,:] .+ s.psf_thresh/2)
  x2_ub = broadcast((a, b) -> a < b ? a : b, im_ub, thetas[2,:] .+ s.psf_thresh/2)

  lb = vec(vcat(x1_lb', x2_lb'))
  #println("lb: ", lb)
  ub = vec(vcat(x1_ub', x2_ub'))
  #println("ub: ", ub)

  p = size(thetas,1)
  su = SupportUpdateProblem(nPoints,p,s,y,w,lossFn)
  function f_and_g!(x,g)
      ps = reshape(x, 2, Int64(length(x)/2))
      ps = vcat(ps, thetas[3:4,:])
      points = reshape(ps,su.p,su.nPoints)
      #println("su.w: ", su.w)
      output = phi(s,points)#s.w)
      residual = su.y .- output
      #l,v_star = loss(s.lossFn,residual)
      l = sum(residual.^2)
      g[:] = computeGradient(s, points, residual)[1:2, :]
      l
  end
  #println("nPoints: ", 2*nPoints)
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

function localDescent_sigma(s :: GaussBlur2D, lossFn :: Loss, thetas ::Matrix{Float64}, w :: Vector{Float64}, y :: Vector{Float64})#, bounds = parameterBounds(s))
  lb,ub = parameterBounds(s)
  #lb, ub = bounds
  s_lb = lb[3]
  s_ub = ub[3]
  w_lb = 0
  w_ub = thetas[4,:].*5


  nPoints = size(thetas,2)

  lb = repeat([s_lb, w_lb], nPoints)
  ub = vec(vcat(fill(s_ub, nPoints)', w_ub'))
  p = size(thetas,1)
  su = SupportUpdateProblem(nPoints,p,s,y,w,lossFn)
  #f_and_g!(x,g) = localDescent_f_and_g!(x,g,su)

  #coordinates are fixed, sigma is input parameter to be optimized
  function f_and_g!(sigma_weight,g)
      crds = thetas[1:2,:]
      sws = reshape(sigma_weight, 2, Int64(length(sigma_weight)/2))
      ps = vcat(crds, sws)

      #results = localDescent_f_and_g!(vec(ps),g,su)
      points = reshape(ps,su.p,su.nPoints)
      output = phi(su.s,points)
      residual = su.y .- output
      #l,v_star = loss(su.lossFn,residual)
      l = sum(residual.^2)
      #grd = computeGradient(s, points[4,:], points, residual)
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
