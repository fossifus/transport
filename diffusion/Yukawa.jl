module Yukawa

    export YukawaSolve, YukawaSolveMatrixFree

    using FFTW
    using LinearAlgebra: norm, dot, I
    using SpecialFunctions: besselk
    using IterativeSolvers: gmres
    using LinearMaps: LinearMap

    function YukawaSolveMatrixFree(dΩ, f, s)

        # geometry
        # n̂, J, κ = Yukawa.GeomDerivs(dΩ)
        n̂ = vcat([d[1] for d in Yukawa.GeomDerivs.(dΩ)]...)
        J = vcat([d[2] for d in Yukawa.GeomDerivs.(dΩ)]...)
        κ = vcat([d[3] for d in Yukawa.GeomDerivs.(dΩ)]...)
        dΩ = vcat(dΩ...)
        # solve SKIE
        σ, gmres_log = gmres(Yukawa.matvec(dΩ, s), f.(dΩ, s); reltol = 1e-12, maxiter=128, log=true)
        @show gmres_log
        # evaluate DLP at (x; s)
        Cʰ(x, s) = -sqrt(s)/length(dΩ) * sum(besselk.(1, sqrt(s) * abs.(x .- dΩ)) .* real(dot.(x .- dΩ, n̂)) ./ abs.(x .- dΩ) .* σ .* J)
        # construct C from Cp, Ch
        C(x, s) = Cʰ(x, s) - f(x, s)
        # C(x, s) = -f(x, s)

        return C

    end

    function YukawaSolve(dΩ, f, s)

        # geometry
        n̂ = vcat([d[1] for d in Yukawa.GeomDerivs.(dΩ.dΩ)]...)
        J = vcat([d[2] for d in Yukawa.GeomDerivs.(dΩ.dΩ)]...)
        κ = vcat([d[3] for d in Yukawa.GeomDerivs.(dΩ.dΩ)]...)
        dΩ′ = vcat(dΩ.dΩ...)

        # solve SKIE
        σ, gmres_log = gmres(I/2 + Yukawa.DLP(dΩ′, dΩ.M, n̂, J, κ, s), -f.(dΩ′, s); reltol = 1e-12, maxiter=128, log=true)

        # evaluate DLP at (x; s)
        Cʰ(x, s) = -sqrt(s) / dΩ.N * dΩ.M * sum(besselk.(1, sqrt(s) * abs.(x .- dΩ′)) .* real(dot.(x .- dΩ′, n̂)) ./ abs.(x .- dΩ′) .* σ .* J)
        # Cʰ(x, s) = -sqrt(s)/length(dΩ) * sum(besselk.(1, sqrt(s) * abs.(x .- dΩ)) .* real(dot.(x .- dΩ, n̂)) ./ abs.(x .- dΩ) .* σ .* J)
        # construct C from Cp, Ch
        C(x, s) = Cʰ(x, s) - f(x, s)

        return C

    end

    # compute normal vector, jacobian, curvature on dΩ
    function GeomDerivs(dΩ::Array{Complex{Float64}, 1})

        # shifted k vector for fourier diff
        k = (N = length(dΩ)) % 2 == 0 ? [0; 1-N/2:N/2-1] : collect(-(N-1)/2:(N-1)/2)
        # functions for fancy FFTs ;)
        ℱ = x -> fftshift(fft(x))
        ∂ = fx -> im * k .* fx
        ℱ⁻¹ = x -> ifft(ifftshift(x))

        # parametrizing dΩ as r(Θ), crunch derivatives in fourier space
        r′ = (ℱ⁻¹ ∘ ∂ ∘ ℱ)(dΩ)
        r′′ = (ℱ⁻¹ ∘ ∂ ∘ ℱ)(r′)
        # jacobian
        J = norm.(r′)
        # unit outward normal
        n̂ = -im * r′ ./ J
        # curvature
        κ = imag(conj.(r′) .* r′′) ./ J.^3

        return n̂, J, κ
    end

    # generate double layer potential matrix
    function DLP(dΩ, M, n̂, J, κ, s::Complex{Float64})

        # DLP off-diagonal terms
        K(i, j) = -sqrt(s) * besselk(1, sqrt(s) * abs(dΩ[i] - dΩ[j])) * real((dot(dΩ[i] - dΩ[j], n̂[j]))) / abs(dΩ[i] - dΩ[j])
        K_diag(i) = 1/2 * κ[i]
        
        return 2π / length(dΩ) * M * [i == j ? 1/2π * complex(K_diag(i)) * J[i] : 1/2π * complex(K(i, j)) * J[j] for i in 1:length(dΩ), j in 1:length(dΩ)]
        # return 2π/N * [i == j ? 1/2π * complex(K_diag(i)) * J[i] : 1/2π * complex(K(i, j)) * J[j] for i in 1:N, j in 1:N]
    end

    function matvec(dΩ::Array{Complex{Float64}, 1}, s::Complex{Float64})
        
        N = length(dΩ)
        n̂, J, κ = GeomDerivs(dΩ)
        
        # matvec function
        function f(σ)
            DLP = complex(zeros(N))
            for i in 1:N
                r = abs.(dΩ[i] .- dΩ)
                rdotn = real((dot.(dΩ[i] .- dΩ, n̂)))
                K = 1/2π * [j == i ? 1/2 * κ[i] : -sqrt(s) * besselk(1, sqrt(s) * r[j]) * rdotn[j] / r[j] for j in 1:N]
                DLP[i] = 2π/N * sum(K .* J .* σ)
            end
            return 1/2 * σ + DLP
        end
        
        # matvec operator
        return LinearMap(f, N);

    end

    function solveSKIE(dΩ, f, s)
        return gmres(-I/2 + DLP(dΩ, s), f.(dΩ, s); reltol = 1e-12, maxiter=20, log=true)
    end

end