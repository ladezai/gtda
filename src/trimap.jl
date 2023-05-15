using ManifoldLearning;
using StatsAPI;

const SWITCH_ITER = 250
const FINAL_MOMENTUM = 0.8
const INIT_MOMENTUM = 0.5
const INCREASE_GAIN = 0.2
const DAMP_GAIN = 0.8
const MIN_GAIN = 0.01

struct Trimap{NN <: ManifoldLearning.AbstractNearestNeighbors, T <: Real} <: ManifoldLearning.NonlinearDimensionalityReduction
    d :: Int
    nearestneighbor :: NN
    proj :: ManifoldLearning.Projection{T} 
end

StatsAPI.fit(::Type{Trimap}, X::AbstractMatrix{T}, maxoutdim :: Integer=2, 
    maxiter :: Integer=400, initialize:: Symbol=:pca, lr :: T = 0.5,
    weight_temp :: T = 0.5, m₁ :: Int = 10, m₂ :: Int = 5, 
    r :: Int = 5, nntype = ManifoldLearning.BruteForce) where {T<:Real} = trimap(X, maxoutdim, maxiter, initialize, lr, weight_temp, m₁, m₂, r, nntype)
StatsAPI.predict(T :: Trimap) = T.proj


@inline function tempered_log(x :: AbstractVector{T}, t :: T) where {T <: Real}
    """
        Implementation of the tempered log_t with temperature t and data x 
        \frac{x^{1-t} - 1}{1-t}.
    """
    if abs(t - 1.0) < 1e-5
        return log.(x)
    else
        return @. (x^(1.0-t) - 1.0) / (1.0 - t)
    end
end


@inline function squared_euclidean_distance(x1 :: AbstractMatrix{T}, 
                        x2 :: AbstractMatrix{T}) where {T <: Real}
    """
        Implements the squared euclidean distance. Note that this is 
        the distance for matrices, i.e. each point is a column and thus 
        we sum over the rows and obtain a vector of distances for each
        column.
    """
    sum(x-> x.^2, x1 .- x2, dims=1) 
end

@inline function squared_euclidean_distance(x1 :: AbstractVector{T}, x2 :: AbstractVector{T}) where {T <: Real}
    """
        Implements the squared euclidean distance but between two vectors. 
        This is basically the mathematical definition of  |x - y|^2_2 for 
        two R^d vectors x,y.
    """
    sum(x -> x.^2, (x1-x2))
end

function squared_euclidean_distance_cmp(x1 :: AbstractVector{T}, x2 :: AbstractVector{T}) where {T <: Real}
    """
        Implements the distance between two vectors' coordinate but don't compute 
        the norm (i.e. we don't sum over the coordinates).
    """
    @fastmath @. (x1-x2)^2    
end 

@inline function rejection_sample(n_sample :: Int, max_int :: Int, rejects :: AbstractArray{Int}) 
    """
        Generate a sample of size `n_sample` in a range `[1, max_int]`
        such that each element of the sample doesn't belong to `rejects`.

        i.e. If n_sample = 3, max_int = 10, rejects = [1,2,3,4,5]
            then a possible outcome are 
            [6,7,8], [6,6,6], [6,7,7], ..., [9,9,10], ...
        
    """
    result = Array{Int,1}(undef, n_sample) 
    rr = 1:max_int
    i  = 1
    @inbounds while i <= n_sample
        j = rand(rr)
        if !(j in rejects)
            result[i] = j
            i += 1
        end
    end

    return result
end

@inline function rejection_sample_without_repetitions(shape :: Pair{Int, Int}, 
    max_int :: Int, rejects :: AbstractArray{Int})
    """
        This does the same as rejection sampling except it rejects also the 
    points on the same column of the results in order to not have a triples of (i, j, j).
    """
    rows, col = shape 
    result = Array{Int, 2}(undef, rows, col)
    for i in 1:col
        rejection_array = rejects |> copy
        for j in 1:rows
            v = rejection_sample(1, max_int, rejection_array) 
            result[j,i] = v[1]
            push!(rejection_array, v[1])
        end
    end

    return result
end

@inline function generate_random_triplets(points :: AbstractMatrix{T}, 
                                    n_points :: Int,
                                    i :: Int, 
                                    r :: Int) where {T <: Real}
    """
    Generate r random points, if the j points are farther than the k points, 
    then switch the indexes.
    """
    random_indexes = rejection_sample_without_repetitions(2 => r, n_points, [i])
    ppp     = repeat([points[i]], r)
    dsts1   = squared_euclidean_distance_cmp(ppp, points[random_indexes[1,:]])
    dsts2   = squared_euclidean_distance_cmp(ppp, points[random_indexes[2,:]])
    # check what distances have to be switched
    change  = dsts1 .> dsts2 
    # swap some parts of the rows
    random_indexes[:,change] = random_indexes[[2,1], change]

    return random_indexes
end

@inline function generate_triplets(points :: AbstractMatrix{T}, 
                nn :: ManifoldLearning.AbstractNearestNeighbors,
                m₁ :: Int = 10, # Number of k-neighbors
                m₂ :: Int = 5,  # Number of non-neighbors sampled at random
                r :: Int = 5,
                weight_temp :: T = 0.5) where {T <: Real}
    # TODO: add docs and
    d, n_points = points |> size
    idx, dsts = ManifoldLearning.knn(nn, points, m₁)
    
    # the nearest neighbors for each point i is given by the 
    # idx[i] (which returns indexes!), so we only have to construct 
    # the triplets in a nice way 
    mm = m₁ * m₂
    mmr = m₁ * m₂ + r
    triplets = zeros(Int, 3, n_points * mmr)
    @inbounds triplets[1, :] = repeat(1:n_points, inner = mmr)
    # Iterate throughout all points and its neighbors

    # @inbouds disallows for boundary checks to improve efficiency
    @inbounds for i in 0:(n_points-1)
        # neighbors points
        neighbors = idx[i+1]
        @inline triplets[2, (i * mmr + 1):(i * mmr + mm)] = repeat(neighbors, inner=m₂)
        # out points 
        out_points = rejection_sample(mm, n_points, vcat(i, neighbors))
        @inline triplets[3, (i * mmr + 1):(i * mmr + mm)] = out_points
        # Sample every node except of the node i, because the node i is already
        # at the ''top row'' 
        random_points = generate_random_triplets(points, n_points, i+1, r) 
        triplets[2:3, (i*mmr + mm + 1) : (i*mmr + mm + r)] = random_points
    end

    ### Here we start generating the weights as a tempered log of the 
    ### sum of 1 + tilde{w}_{ijk} - w_min, note that this weights are 
    ### depending only on the initial embedding of the data and not 
    ### on the new dimensionally reduced embedding.
    pointsI = points[:, triplets[1,:]]
    pointsJ = points[:, triplets[2,:]]
    pointsK = points[:, triplets[3,:]]

    # compute significance to enchance the analysis by density of close elements
    #OLD CODE: sig = [sqrt(sum(dst[4:6])/3) for dst in dsts]
    sig = map(dd -> sum(x->/(x,3), dd[4:6]), dsts)
    sigI = sig[triplets[1,:]]
    sigJ = sig[triplets[2,:]]
    sigK = sig[triplets[3,:]]

    # distances normalized by the significance 
    dij = vec(squared_euclidean_distance(pointsI, pointsJ)) ./ (sigI .* sigJ)
    dik = vec(squared_euclidean_distance(pointsI, pointsK)) ./ (sigI .* sigK)

    # take the tempered log of the distances to compute the weights.
    weights = dik - dij
    weights .-= minimum(weights)
    weights = tempered_log(1 .+ weights, weight_temp)

    return triplets, weights
end

@inline function trimap_metric(embedding :: AbstractMatrix{T}, 
    triplets :: AbstractMatrix{Int},
    weights :: AbstractVector{T}) where {T <: Real}

    d, n = size(triplets)
    # evaluate the distances
    @inbounds sim_distances = 1.0 .+ squared_euclidean_distance(embedding[:, triplets[1, :]], embedding[:, triplets[2, :]])
    @inbounds out_distances = 1.0 .+ squared_euclidean_distance(embedding[:, triplets[1, :]], embedding[:, triplets[3, :]])

    return @. $(sum)(weights / (1.0 + out_distances / sim_distances)) / n
end

function squared_euclidean_dst_deriv(i :: Int, 
        y_i :: AbstractVector{T}, 
        y_j :: AbstractVector{T}) where {T<: Real}
    return 2 .* (y_i .- y_j) .* ifelse(i%2 == 0, 1, -1) 
end

@inline function local_trimap_grad_i(i :: Int, triple_of_points :: AbstractMatrix{T}) where {T <: Real}
    """
        Compute the gradient for the trimap loss function in case of a single point / triple.
        
        BUG: this is not correct as it overstimates the loss function. The error could be either 
            here or in the weights or in the computation of the triplets.
    """
    dij = squared_euclidean_distance(triple_of_points[:,1], triple_of_points[:,2])
    dik = squared_euclidean_distance(triple_of_points[:,1], triple_of_points[:,3])

    den = (dij + dik + 2.0)^2 
    
    # l_{ijk} derived in y_i
    if i == 1
        ddij = squared_euclidean_dst_deriv(1, triple_of_points[:,1], triple_of_points[:,2])
        ddik = squared_euclidean_dst_deriv(1, triple_of_points[:,1], triple_of_points[:,3])
        return @. ((1+dij) * ddik - (1+dik) * ddij) / den
    # l_{ijk} derived in y_j
    elseif i == 2
        ddij = squared_euclidean_dst_deriv(2, triple_of_points[:,1], triple_of_points[:,2])
        return @. -((1+dik) * ddij / den)
    # l_{ijk} derived in y_k
    elseif i == 3
        ddik = squared_euclidean_dst_deriv(2, triple_of_points[:,1], triple_of_points[:,3])
        return @. (1+dij) * ddik / den
    else
        error("This can't happen, triplets consists of 3 elements only!")
    end
end

@inline function trimap_loss_grad!(grad :: AbstractMatrix{T}, 
    embedding :: AbstractMatrix{T},
    triplets :: AbstractMatrix{Int},
    weights :: AbstractVector{T}) where {T <: Real}
    """
        BUG: With this instead of the previous definition we have far better 
        performance improvements, but it is still not corrected implemented!
        
        BUG: It is almost right, it has some rough edges for some datapoint,
        but I don't know why (maybe the weights? maybe the triplet sampling?)
    """
    
    d, n_points = size(embedding)
    grad .= 0
    #
    #for p in 1:n_points
        ## Gives a vector of cartesian indexes
        #axis = findall(==(p), triplets)
        ## note that this goes down by rows and then by cols
        #for i in axis
            ##println(i)
            #triple = embedding[:, triplets[:, i[2]]]
            ##@assert(size(triple)[2] == 3)
            ##@assert(size(triple)[1] == 2)
            #vv = local_trimap_grad_i(i[1], triple)
            ##@assert(sum(abs.(vv)) < 10^6) # BUG: THE VV GO TO tHE MOON!
            #grad[:, p]  .+= vv .* weights[i[2]]
        #end
    #end

    # NOTE: This is very fast, but still to figure out if it is 
    # completely right. 
    r, n_triplets = size(triplets)
    @simd for t in 1:n_triplets 
        @inbounds triple_points = embedding[:,triplets[:,t]]
        @simd for i in 1:r
            gradi = local_trimap_grad_i(i, triple_points)
            @inbounds grad[:,triplets[i,t]] .+= gradi .* weights[t]
        end
    end
        
end

@fastmath @inline function update_embedding!(embedding, grad, vel, gain, lr, itr)
    """
        grad: gradient
        gain: parameter for the optimization 
        vel: velocity/speed for the optimization
        lr : learning rate
        itr: iteration number
    """
    # Implements the delta-bar-delta method for optimizing 
    # the embedding
    gamma = ifelse(itr > SWITCH_ITER, FINAL_MOMENTUM, INIT_MOMENTUM)

    # optimizer via delta-bar-delta method
    check          = @. sign(vel) != sign(grad)
    @. gain[check] = gain[check] + INCREASE_GAIN
    ##  TODO: check if we can remove some dots and/or remove fusing
    ##gain[.~check] = max.(gain[.~check] .* DAMP_GAIN, MIN_GAIN) 
    @. gain[~check] = max(gain[~check] * DAMP_GAIN, MIN_GAIN)
    @. vel = gamma * vel - lr * gain * grad 
    embedding .+= vel 
end


@inline function trimap(X :: AbstractMatrix{T}, 
                maxoutdim :: Integer=2, 
                maxiter :: Integer=400, 
                initialize:: Symbol=:pca,
                lr :: T = 0.5,
                weight_temp :: T = 0.5,
                m₁ :: Int = 10,
                m₂ :: Int = 5,
                r :: Int = 5,
                nntype = ManifoldLearning.BruteForce) where {T <: Real}
    """
        Implements the routine as described in 
         https://arxiv.org/abs/1910.00204
        
    """
    d, n = size(X)

    Y = if initialize == :pca 
            predict(fit(ManifoldLearning.PCA, X, maxoutdim=maxoutdim), X)
        elseif initialize == :random
            rand(T, maxoutdim, n)
        else error("Unknown initialization")
    end

    # Neareest neighbors
    NN = fit(nntype, X) 
    
    # initialize triplets and weights
    triplets, weights = generate_triplets(X, NN, m₁, m₂, r, weight_temp)

    # Optimization of the embedding
    gain = zeros(T, size(Y))
    vel  = zeros(T, size(Y))
    grad = zeros(T, size(Y))
    #embedding_gradient!(y) = (grad .= Zygote.gradient(emb -> trimap_metric(emb, triplets, weights), y)[1])
    @inbounds for i in 1:maxiter
        gamma = ifelse(i > SWITCH_ITER, FINAL_MOMENTUM, INIT_MOMENTUM)

        # TODO: THE AUTOMATIC GRADIENT EVALUATION IS THE PERFORMANCE ISSUE!
        #grad = gradient(fnLos, Y .+ gamma .* vel)[1]
        # Still slow even with forward diff..
        #embedding_gradient!(Y .+ gamma .* vel)
        # do everything in-place
        trimap_loss_grad!(grad, Y .+ gamma .* vel, triplets, weights)
        update_embedding!(Y, grad, vel, gain, lr, i)
    end

    return Trimap{nntype,T}(d, NN, Y)
end


