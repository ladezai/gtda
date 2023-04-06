using ManifoldLearning
using Ripserer
using MultivariateStats
using PersistenceDiagrams
using Random
using Plots


function plot_homology(T, data, name)    
    # use the some embeddings to compute the distances
    if T == MultivariateStats.PCA
        model = fit(T, data; maxoutdim=2)
        pred = predict(model, data)' 
    else
        model = fit(T, data)
        pred = predict(model)'
    end
    
    # Ripserer wants tuples as points not matrices
    formattedData = [Tuple(r) for r in eachrow(pred)]
    # Display the plots
    reducedDataVis = scatter(formattedData)
    display(reducedDataVis)
    readline()
    # save it
    savefig(reducedDataVis, "reduced_$name.png") 
    # compute the alpha complex persistence diagrams
    persistency_hom = plot(ripserer(Rips(formattedData); modulus=2))
    # visualize and save the figure
    display(persistency_hom)
    readline()
    savefig(persistency_hom, "per_hom_$name.png")
end

# List all the functions we want to test
embs = Dict("Isomap" => ManifoldLearning.Isomap, "LLE" => ManifoldLearning.LLE,
            "PCA" => MultivariateStats.PCA)
# set some seed
Random.seed!(1234)
# Generate the swiss roll
X, L = ManifoldLearning.swiss_roll(1000)

# Plot for every embedding 
for (key, val) in embs
    plot_homology(val, X, "$(key)_1000")
end

# same stuff but with less data points
Random.seed!(12341)
X, L  = ManifoldLearning.swiss_roll(500)
for (key, val) in embs
    plot_homology(val, X, "$(key)_500")
end
