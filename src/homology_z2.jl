
"""
    A small script to compute the simplex homology up to
    dimension 2 in 𝚭₂ using BitArray for memory efficiency. 
    (although few things in here are truly efficient...)
"""


struct ComplexSimplex
    F_p       :: Set{Set{Integer}}
    dimension :: Integer
end 


function generate_all_subsets_smaller_order(face :: Set{Integer}) :: Set{Set{Integer}}
    """
        Given a face σ of dimension d, returns all the subsets of order d-1.
        
        For example if we consider σ = [1,2,3] it will return 
        [1,2],[2,3],[1,3].

        Note: this operation is linear.
    """
    return Set(map(v -> filter(!=(v), face), collect(face)))
end

function generate_from_maximal_faces(complex :: ComplexSimplex) :: ComplexSimplex
    """
        Returns a new complex simplex such that has the same 
        maximal faces of the original, but it is consistent with the definition 
        of Complex Simplex, i.e. it is closed under subsets.

        Note that this operation is ∑ |F_i| ⋅ i 
    """
    d           = complex.dimension
    completeF_p = deepcopy(complex.F_p)
    # First iterate over all dimensions.
    for i in d:-1:1
        # Iterate over all faces in a dimension
        for face in completeF_p
            # If its the previously generated level of faces,
            # compute the next one
            if length(face) == i+1
                new_faces = generate_all_subsets_smaller_order(face)
                union!(completeF_p, new_faces)
            end
        end     
    end
    
    # Finally return a new Complex Simplex
    return ComplexSimplex(completeF_p, d)
end

function faces_of_dimension_p(complex :: ComplexSimplex, p :: Integer) :: Set{Set{Integer}}
    """
        Returns the faces of dimension p from a complex simplex.
        Note that the dimension p is equal to the cardinality of a face -1, 
        i.e. p = #σ - 1, with σ being a face of a complex simplex.

        Returns the empty set if p > dimension of the complex simplex.

        Note that this operation is linear, so it should be avoided.
    """
    if p > complex.dimension
        return Set()
    end
    faces = filter(x -> length(x) == p+1, complex.F_p) 
    return faces 
end


function _gaussian_elim_z2_pivot(mat :: BitArray{2}) :: BitArray{2} 
    """
        This routine computes the gaussian elimination algorithm with pivoting
        in 𝐙₂.

        Note that the result will be a upper triangular matrix, which is used only
        to compute the rank.
    """
    row, col = size(mat)
    
    for i in 1:(row-1)
        if col < i
            break 
        end

        # pivot if no 1 on the diagonal
        if ~mat[i,i]
            for j in (i+1):row
                if mat[j,i] == 1 
                    tmp      = copy(mat[i,:])
                    mat[i,:] = copy(mat[j,:])
                    mat[j,:] = tmp 
                    break
                end
            end
        end
        
        # If no 1 found, skip the row
        if ~mat[i,i]
            continue 
        end
        
        # In boolean arithmetics addition is given by xor. 
        # So, when we do the elimination by row, we have to 
        # compute the xor by element i.e. `xor.`
        for j in (i+1):row
            if mat[j,i]
                mat[j,:] = xor.(mat[j,:], mat[i,:])
            end
        end
    end
    
    # removes also the columns not really used 
    #for i in 1:(col-1)
        #if row < i
            ###println("not completed reduction because rows are smaller! $i")
            #break
        #end
        #if mat[i,i]
            #mat[i,(i+1):end] .= 0
        #end
    #end

    return mat
end

function rank(mat :: BitArray{2}) :: Int
    """
        Computes the row rank of a BitArray
    """
    m = 0
    try 
        m = reduce(+, map(r -> 1 ∈ r, eachrow(mat)))
    catch e  # In case there are no ones, get 0.
        m = 0
    end
    return m
end

function _generate_boundary_matrix(C_k :: Set{Set{Integer}}, C_km1 :: Set{Set{Integer}}) :: BitArray{2}
    """
        Given the generator set of C_k and the generators of C_{k-1} it constructs a matrix that
        maps the elements from C_k to C_{k-1} under the given basis of elements (note that this matrix 
        is in 𝐙₂).
    """ 
    
    # Get an ordered version of the sets, because we have to fix 
    # a basis and thus an ordering.
    C_kA   = collect(C_k)
    C_km1A = collect(C_km1)
    mat :: BitArray{2} = zeros(length(C_km1A), length(C_kA))
    # Iterate throughout the basis C_ka, then fill the matrix by columns
    for (i, gen) in enumerate(C_kA)
        subsets :: Set{Set{Int}}  = generate_all_subsets_smaller_order(gen)
        indexes :: Array{Int}     = findall(z -> z ∈ subsets, C_km1A)
        mat[:,i]                  = map(i -> i ∈ indexes, 1:length(C_km1A))
    end
    return mat
end

"""
    To verify if the gaussian elimination works,
    here are some tests.
"""
function _test_gaussian_elim()
    println("Test gaussian elimination!")
    b = BitArray{2}([1 0 0 0 1; 1 0 1 1 0; 0 1 0 1 1])
    c = _gaussian_elim_z2_pivot(copy(b))
    correct = BitArray{2}([1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0])
    if c != correct
        println("Matrix is \n$b\neliminated is\n$c")
    end
    b = BitArray{2}([1 1 0; 0 1 1; 1 0 1])
    c = _gaussian_elim_z2_pivot(copy(b))
    correct = BitArray{2}([1 0 0; 0 1 0; 0 0 0])
    if c != correct
        println("Matrix is \n$b\neliminated is\n$c")
    end
    println("End test gaussian elimination")
end
#_test_gaussian_elim()

# Get all the faces by dimension
#C_3 = faces_of_dimension_p(newComplex, 3)
#
function compute_betty_numbers(complex)
    """
        Computes the betty numbers (not reduced), of a complex in 2 dimensions.
    """
    C_2 = faces_of_dimension_p(complex, 2)
    C_1 = faces_of_dimension_p(complex, 1)
    C_0 = faces_of_dimension_p(complex, 0)
    n_2 = length(C_2)
    n_1 = length(C_1)
    n_0 = length(C_0)
    
    # Ok theoretically it is ok, but I need 
    ∂₂ = _generate_boundary_matrix(C_2, C_1)
    #println(∂₂)
    ∂₂ = _gaussian_elim_z2_pivot(∂₂)
    #println(∂₂)
    r_2    = rank(∂₂) 
    #println(r_2)
    ∂₁ = _generate_boundary_matrix(C_1, C_0)
    #println(∂₁)
    ∂₁ = _gaussian_elim_z2_pivot(∂₁)
    #println(∂₁)
    r_1    = rank(∂₁)
    #println(r_1)
    z_2    = n_2 - r_2 
    z_1    = n_1 - r_1
    return [z_2, z_1-r_2, 1]
end

function compute_examples()
    # Start defining the maximal faces
    complex1 = ComplexSimplex(Set([Set([1, 2, 3])]), 2)
    # Construct the full simplex
    newComplex = generate_from_maximal_faces(complex1)
    # Check if the betties number are right
    betties = compute_betty_numbers(newComplex)
    println("The betty numbers of\n$newComplex, are\n$betties")
    
    complex1 = ComplexSimplex(Set([Set([1, 2, 3]), Set([2,4]), Set([1,4])]), 2)
    newComplex= generate_from_maximal_faces(complex1)
    betties = compute_betty_numbers(newComplex)
    println("The betty numbers of\n$newComplex, are\n$betties")

    complex1 = ComplexSimplex(Set([Set([1, 2, 3]), Set([4,2]), Set([4,1])]), 2)
    newComplex = generate_from_maximal_faces(complex1)
    betties = compute_betty_numbers(newComplex)
    println("The betty numbers of\n$newComplex, are\n$betties")
end

function compute_torus()
    b :: Array{Array{Int}} = [[1, 2, 4],[2, 4,  5],[2, 3, 5],[3, 5, 6],[1, 3, 4],[4, 6, 3],
             [5, 6, 7],[7, 8,  5],[8, 9, 5],[4, 5, 9],[6, 4, 7],[7, 9, 4],
             [8, 9,11],[11,12, 9],[8,10,11],[7, 8,10],[10,12,9],[7, 9,10],  
             [1, 2,10],[2, 11,10],[2, 3,12],[2,12,11],[1, 3,10],[3,12,10]]
    b1 :: Array{Set{Int}}  = map(Set, b)
    torus = ComplexSimplex(Set(b1), 2)
    completeTorus :: ComplexSimplex = generate_from_maximal_faces(torus)
    betties = compute_betty_numbers(completeTorus)
    println("The betty numbers of the Torus, are $betties")
    # correct result
    # 1 2 0 
end

#compute_examples()
compute_torus()
