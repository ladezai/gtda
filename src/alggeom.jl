using AbstractAlgebra;
using Groebner;
using Latexify;


function printLatexCode(system_of_polynomials)
    str = ""
    for p in system_of_polynomials
        str *= "& " * replace(latexify(repr(p)), ("\$"=> "")) * " = 0 \\" * "\\ \n"
    end
    str |> print
end

function exercise1()
    """
    Find a Gröbner basis for the ideal (y-x^3, z-x^2) in the 
    lexicographic order.
    """
    R, (x,y,z) = PolynomialRing(QQ, ["x","y", "z"]);
    polys = [y-x^3, z-x^2];
    G = groebner(polys);
    printLatexCode(G)
end

function exercise2()
    """
    Builds equations for a process Z that is a mixed binomial y B(4, h) +
    (1-y) B(4, t) where y is the ``mixing'' variable, h the bias of the first
    coin and t the bias of the second coin.
    
    It also computes and print the associated grobner basis.
    """
    R, (y,h,t, z₀, z₁, z₂, z₃, z₄) = PolynomialRing(QQ, ["y", "h", "t", "z_0", "z_1", "z_2", "z_3", "z_4"]);
    zs = [z₀, z₁, z₂, z₃, z₄]
    @fastmath binom(k) = 24 / (factorial(4 - k) * factorial(k))
    mixedBinomial(k) = binom(k) * (y * h^k * (1-h)^(4-k) + (1-y) * t^k * (1-t)^(4-k))

    polys = [mixedBinomial(k) - zs[k+1] for k in 0:4];
    printLatexCode(polys)
    #G = groebner(polys)

    G = groebner(polys);
    printLatexCode(G)
end
