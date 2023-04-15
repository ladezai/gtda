using ManifoldLearning
using Ripserer
using MultivariateStats
using PersistenceDiagrams
using Random
using Plots


function plot_homology(T, data, name, save_and_show)    
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
    if save_and_show
        reducedDataVis = scatter(formattedData)
        display(reducedDataVis)
        readline()
        # save it
        savefig(reducedDataVis, "reduced_$name.png") 
    end
    # compute the alpha complex persistence diagrams
    persistency_hom = ripserer(Rips(formattedData); modulus=2)
    # visualize and save the figure
    if save_and_show
        fig = plot(persistency_hom)
        display(fig)
        readline()
    end
    #savefig(fig, "per_hom_$name.png")
    return persistency_hom
end


function main()
    # List all the functions we want to test
    embs = Dict("Isomap" => ManifoldLearning.Isomap, "LLE" => ManifoldLearning.LLE,
                "PCA" => MultivariateStats.PCA)

    per_homs = Dict("Isomap" => [PersistenceDiagram([])], "LLE" => [PersistenceDiagram([])], 
        "PCA" => [PersistenceDiagram([])])
    # set some seed
    Random.seed!(1234)
    # Generate the swiss roll
    X, L = ManifoldLearning.swiss_roll(1000)

    # Plot for every embedding 
    for (key, val) in embs
        phm = plot_homology(val, X, "$(key)_1000", false)
        per_homs[key]= phm
    end

    j = 0 
    for (key1, val1) in per_homs
        for (key2, val2) in per_homs
            if key1 == key2
                continue
            end
            dst = Bottleneck()(val1, val2)
            println("Bottleneck distance between $key1 and $key2: $dst")
        end
        # compute only the off diagonal distances
        j = j + 1 
        if j > 1
            break
        end
    end

    # same stuff but with less data points
    Random.seed!(12341)
    X, L  = ManifoldLearning.swiss_roll(500)
    for (key, val) in embs
        plot_homology(val, X, "$(key)_500", false)
    end
end


main()
