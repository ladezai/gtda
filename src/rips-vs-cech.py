import numpy as np
import matplotlib.pyplot as plt
import gudhi


# Generate the initial data
nonprimes_below12 = np.array([1,4,6,8,9,10,12])
sphere_without_primes = np.array([np.cos(nonprimes_below12 * 2 * np.pi / 12),
                                     np.sin(nonprimes_below12 * 2 * np.pi /
                                                12)]).T

"""
    The following code provides the barcodes for the persistence homology of
    the Vietoris-Rips complex of the given data. As one can see from the 
    picture, as the radious increases we have the formation of a single
    connected component (long red bar) and the creation of a 1 dimensional hole
    and at last its disappearence (blue bottom bar). We can see that its
    construction is slower (we need larger radii in order to get the whole 
    evolution of the persistence barcode).
"""
rips         = gudhi.RipsComplex(points=sphere_without_primes,
                                 max_edge_length=3) 
rips_simplex_tree = rips.create_simplex_tree(max_dimension=2)
rips_diag    = rips_simplex_tree.persistence()
gudhi.plot_persistence_barcode(rips_diag, legend=True)
plt.show()
gudhi.plot_persistence_diagram(rips_diag, legend=True)
plt.show()
# visualize persistence diagram

"""
    To test the Cech complex behaviour, since neither Gudhi nor 
    Ripserer provide Cech complexes, we are using Alpha complexes 
    due to the fact that they should be at least homeomorphic (and 
    as the barcodes below picture), we don't have much differences 
    from the expected behaviour (i.e. the 1-dim hole disappears
    as the radious is larger than 1 and it appears at about 1/2 which
    is close to sqrt(2)/2). 
"""
cech         = gudhi.AlphaComplex(points=sphere_without_primes)
cech_simplex_tree = cech.create_simplex_tree()
cech_diag    = cech_simplex_tree.persistence()
gudhi.plot_persistence_barcode(cech_diag, legend=True)
plt.show()
gudhi.plot_persistence_diagram(cech_diag, legend=True)
plt.show()
# visualize persistence diagram


