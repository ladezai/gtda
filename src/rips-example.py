import numpy as np
import matplotlib.pyplot as plt
import gudhi


# Generate the initial data
nonprimes_below12 = np.array([1,4,6,8,9,10,12])
sphere_without_primes = np.array([np.cos(nonprimes_below12 * 2 * np.pi / 12),
                                     np.sin(nonprimes_below12 * 2 * np.pi /
                                                12)]).T
N = 100
nn = np.arange(N)
sphere = np.array([np.cos(nn * 2 * np.pi / N),
                   np.sin(nn * 2 * np.pi / N)]).T

# Plot the data, plus a circle.
plt.plot(sphere[:,0], sphere[:,1], color="gray")
plt.scatter(sphere_without_primes[:,0], sphere_without_primes[:,1], color="b")
plt.ylim([-1.5,1.5])
plt.xlim([-1.5,1.5])
plt.title(r'$\mathbf{S}^1$')
plt.show()

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

# visualize persistence diagram and barcode
gudhi.plot_persistence_barcode(rips_diag, legend=True)
plt.show()
gudhi.plot_persistence_diagram(rips_diag, legend=True)
plt.show()


