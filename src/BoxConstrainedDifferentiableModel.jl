export BoxConstrainedDifferentiableModel
using NLopt
# A simple forward model with box constrained parameters.
# Assumes differentiablity of the forward operator.
#
# For concrete examples see examples/smi or examples/sysid.
#
abstract type BoxConstrainedDifferentiableModel <: ForwardModel end

# Sets the parameters for the continuous optimizer.
# Can be overwridden.
function initializeOptimizer!(model :: BoxConstrainedDifferentiableModel, opt :: Opt)
  ftol_abs!(opt, 1e-6)
  xtol_rel!(opt, 0.0)
  maxeval!(opt, 200)
end

# Default implementation. Uses NLopt to do continuous optimization.


struct SupportUpdateProblem
  nPoints :: Int64
  p :: Int64
  s :: BoxConstrainedDifferentiableModel
  y :: Vector{Float64}
  w :: Vector{Float64}
  lossFn :: Loss
end

# Default implementation. Uses NLopt to do continuous optimization.
function localDescent(s :: BoxConstrainedDifferentiableModel, lossFn :: Loss, thetas ::Matrix{Float64}, w :: Vector{Float64}, y :: Vector{Float64}, bounds = parameterBounds(s))
  #lb,ub = parameterBounds(s)
  lb, ub = bounds
  nPoints = size(thetas,2)
  p = size(thetas,1)
  su = SupportUpdateProblem(nPoints,p,s,y,w,lossFn)
  f_and_g!(x,g) = localDescent_f_and_g!(x,g,su)
  opt = Opt(NLopt.LD_MMA, length(thetas))
  initializeOptimizer!(s, opt)
  min_objective!(opt, f_and_g!)
  #lower_bounds!(opt, vec(repmat(lb,1,nPoints)))
  #upper_bounds!(opt, vec(repmat(ub,1,nPoints)))
  lb_vec = vec(repeat(lb,1,nPoints))
  ub_vec = vec(repeat(ub,1,nPoints))
  #println("LD lb_vec: ", lb_vec)
  #println("LD ub_vec: ", ub_vec)
  lower_bounds!(opt, lb_vec)
  upper_bounds!(opt, ub_vec)
  (optf,optx,ret) = optimize(opt, vec(thetas))
  #println("ret: ", ret)
  return reshape(optx,p,nPoints), optf
end

function localDescent_f_and_g!(points :: Vector{Float64}, gradient_storage :: Vector{Float64}, s :: SupportUpdateProblem)
  points = reshape(points,s.p,s.nPoints)
  output = phi(s.s,points)
  residual = s.y .- output
  l,v_star = loss(s.lossFn,residual)
  gradient_storage[:] = computeGradient(s.s, points, residual)
  return l
end
