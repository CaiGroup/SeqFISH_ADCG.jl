export ADCG

function ADCG(sim :: ForwardModel, lossFn :: Loss, y :: Vector{Float64}, tau :: Float64, min_weight :: Float64;
  match_ϵ :: Float64 = 0.1,
  callback :: Function = (old_thetas,thetas,output,old_obj_val) -> false,
  max_iters :: Int64 = 50,
  max_cd_iters :: Int64 = 200)


  @assert tau > 0.0
  bound = -Inf
  thetas = zeros(0,0)

  #weights = zeros(0)
  #cache the forward model applied to the current measure.
  output = zeros(length(y))
  dot_record = initialize_dot_records(sim, thetas)
  for iter = 1:max_iters
    #compute the current residual
    residual = y .- output
    #evalute the objective value and gradient of the loss
    objective_value, grad = loss(lossFn, residual)
    #compute the next parameter value to add to the support
    theta,score = lmo(sim,residual)#grad)
    #score is - |<\psi(theta), gradient>|
    #update the lower bound on the optimal value
    bound = max(bound, objective_value+score*tau-dot(output,grad))
    #check if the bound is met.

    #if(objective_value < bound + min_optimality_gap || score >= 0.0)
      #println("bound: ", bound)
      #println("objective_value: ", objective_value)
      #println("score: ", score)
      #println("theta: ", theta)
      #return thetas,weights
    #end
    #update the support
    old_thetas = thetas
    thetas = iter == 1 ? reshape(theta, length(theta),1) : [thetas theta]
    #run local optimization over the support.
    #old_weights = copy(weights)
    thetas = localUpdate(sim,lossFn,thetas,y,tau,max_cd_iters, min_weight)
    output = phi(sim, thetas)
    update_records!(sim, dot_record, thetas, match_ϵ, sim.dims)
    #println("record update:")
    #println(dot_record)
    if callback(old_thetas, thetas, output, objective_value)
      #println("return from callback")
      return dot_record
      #return old_thetas
    end
  end
  println("Hit max iters in frank-wolfe!")
  return dot_record
  #return thetas
end

function localUpdate(sim :: ForwardModel,lossFn :: Loss,
    thetas :: Matrix{Float64}, y :: Vector{Float64}, tau :: Float64, max_iters, min_weight)
  w_ind = size(thetas)[1]

  #println("thetas start:")
  #println(thetas)
  close_pnts, far_pnts = get_close_network(sim, thetas)
  if length(close_pnts) == 1
    #println("close points 1: return")
    return thetas
  end
  all_thetas = thetas
  far_thetas = thetas[:, far_pnts]
  thetas = thetas[:, close_pnts]

  #println("close thetas:")
  #println(thetas)

  removed_points = []

  y .-= phi(sim, far_thetas)

  for cd_iter = 1:max_iters
    old_thetas = thetas
    #remove points that are too close together
    #ToDo, make sure not to place new dot where dots have been removed
    #thetas, weights = remove_duplicates(thetas, weights, y, sim.sigma_lb, sim.sigma_ub, tau, 0.0, sim.sigma_lb)


    new_sigmas, optf = localDescent_sigma(sim, lossFn, thetas, thetas[w_ind,:], y)

    if typeof(sim) == GaussBlur2D
      thetas[3:4,:] = new_sigmas
    elseif typeof(sim) == GaussBlur3D
      thetas[4:6,:] = new_sigmas
    else
      error("Unknown Source funtion type")
    end

    #remove points with low weight
    #Need to keep track of this in close network!
    if any(thetas[w_ind,:].<= min_weight)
      #println("Removing ", close_pnts[thetas[w_ind,:] .<= min_weight], " dim points. min_weight: $min_weight")
      #println(thetas)
      #println()
      
      removed_points = vcat(removed_points, close_pnts[thetas[w_ind,:] .<= min_weight])
      close_pnts = close_pnts[thetas[w_ind,:] .> min_weight]
      thetas = thetas[:,thetas[w_ind,:] .> min_weight]
    end
    #local minimization over the support
    new_coords, optf = localDescent_coord(sim, lossFn, thetas, thetas[w_ind,:], y)
    new_sigmas, optf = localDescent_sigma(sim, lossFn, thetas, thetas[w_ind,:], y)
    new_thetas = vcat(new_coords, new_sigmas)
    #break if termination condition is met
    #if length(thetas) == 0 || (length(thetas) == length(new_thetas) && maximum(abs.(vec(thetas)-vec(new_thetas))) <= 1E-7)
    if length(thetas) == 0
        #println("No sources found")
        break
    elseif (length(thetas) == length(new_thetas) && maximum(abs.(vec(thetas)-vec(new_thetas))) <= 1E-7)
        #println("No iteration found no change")
        break
    end
    thetas = new_thetas
  end

  #println("pre prune thetas:")
  #println(thetas)
  #final prune
  if any(thetas[w_ind,:].<= min_weight)
    #println("Removing ",sum(thetas[4,:] .<= min_weight), " dim points.")
    #vcat(removed_points, close_points(thetas[w_ind,:] .<= min_weight))
    close_pnts = close_pnts[thetas[w_ind,:] .> min_weight]
    thetas = thetas[:,thetas[w_ind,:] .> min_weight]
  end

  #println("thetas:")
  #println(thetas)
  #println("close_pnts: ", close_pnts)
  all_thetas[:, close_pnts] .= thetas
  #all_thetas = all_thetas[:, close_pnts] .= thetas
  remaining_points = sort(vcat(far_pnts, close_pnts))
  #println("atr: ")
  #println(all_thetas)
  #println("remaining_points: $remaining_points")
  return all_thetas[:, remaining_points] #thetas
end
