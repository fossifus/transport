using Plots
using MATLAB

# includet("Yukawa.jl")
includet("Talbot.jl")

# using .Yukawa
using .Talbot

####################################
# ADD THIS SECTION TO DIFFUSION.JL #
####################################

using FFTW
using LinearAlgebra: norm, I, dot
using SpecialFunctions: besselk, besselj0, bessely0
using IterativeSolvers: gmres    

struct Boundary
    z::Array{Complex{Float64}, 1}
    M::Integer # number of bodies
    N::Integer # discretization pts per body
    n̂::Array{Complex{Float64},1}; J::Array{Float64,1}; κ::Array{Float64,1} # geom derivs
    f::Function # boundary condition

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

    # constructor
    function Boundary(dΩ::Array{Array{Complex{Float64}, 1}, 1}; bc::Symbol)
        # select boundary condition function
        if bc == :heat
            f = (x, s) -> -1.0/s
        elseif bc == :pointsource
            f = (x, s) -> sum([1/2π * besselk(0, sqrt(s) * norm(x - x₀)) for x₀ in x₀])
        end
        # compute derivatives on boundary
        n̂ = vcat([d[1] for d in GeomDerivs.(dΩ)]...)
        J = vcat([d[2] for d in GeomDerivs.(dΩ)]...)
        κ = vcat([d[3] for d in GeomDerivs.(dΩ)]...)
        # reshape input array
        z = vcat(dΩ...); M = length(dΩ); N = length(dΩ[begin])
        return new(z, M, N, n̂, J, κ, f)
    end
end


##############################

# generate double layer potential matrix
function DLP(dΩ::Boundary, s::Complex{Float64})

    # DLP off-diagonal terms
    K(i, j) = -sqrt(s) * besselk(1, sqrt(s) * abs(dΩ.z[i] - dΩ.z[j])) * real((dot(dΩ.z[i] - dΩ.z[j], dΩ.n̂[j]))) / abs(dΩ.z[i] - dΩ.z[j])
    K_diag(i) = 1/2 * dΩ.κ[i]
    
    return 2π / dΩ.N * [i == j ? 1/2π * complex(K_diag(i)) * dΩ.J[i] : 1/2π * complex(K(i, j)) * dΩ.J[j] for i in 1:length(dΩ.z), j in 1:length(dΩ.z)]
end

function pointwise_flux(dΩ::Boundary; s::Complex{Float64}, x₀::Array{Complex{Float64},1})
    σ, gmres_log = gmres(-I/2 + DLP(dΩ, s), -dΩ.f.(dΩ.z, s); reltol = 1e-12, maxiter=128, log=true);
    
    x = [real(dΩ.z) imag(dΩ.z)]
    K1(i, j) = besselk(1, sqrt(s) * norm(x[i,:] - x[j,:]))
    K2(i, j) = norm(x[i,:] - x[j,:])
    K3(i, j) = dot(x[i,:] - x[j,:], [real(dΩ.n̂[j]) imag(dΩ.n̂[j])])
    dK1(i, j) = sqrt(s) * (-besselk(0, sqrt(s) * norm(x[i,:] - x[j,:])) - besselk(1, sqrt(s) * norm(x[i,:] - x[j,:])) / norm(x[i,:] - x[j,:]) / sqrt(s)) * (x[i,:] - x[j,:])' / norm(x[i,:] - x[j,:])
    dK2(i, j) = (x[i,:] - x[j,:])' / norm(x[i,:] - x[j,:])
    dK3(i, j) = [real(dΩ.n̂[j]) imag(dΩ.n̂[j])]
    K(i, j) = sqrt(s) / 2π * dot([real(dΩ.n̂[i]) imag(dΩ.n̂[i])], (dK1(i, j) * K2(i, j) - K1(i, j) * dK2(i, j)) * K3(i, j) / K2(i, j) / K2(i, j) + K1(i, j) * dK3(i, j) / K2(i, j))

    # (K - λ) term
    dxdylog(i, j) = -real(dot(dΩ.n̂[i], dΩ.n̂[j])) / abs(dΩ.z[i] - dΩ.z[j])^2 + 2(real(dot(dΩ.z[i] - dΩ.z[j], dΩ.n̂[i])) * real(dot(dΩ.z[i] - dΩ.z[j], dΩ.n̂[j]))) / abs(dΩ.z[i] - dΩ.z[j])^4
    λ = -1 / 2π
    oddeven1(i) = sum([(K(i,j) - λ * dxdylog(i, j)) * σ[j] * dΩ.J[j] * 2 * 2π / length(dΩ.z) for j in (iseven(i) ? (1:2:length(dΩ.z)) : (2:2:length(dΩ.z)))])

    # + λ term
    oddeven2(i) = sum([λ * dxdylog(i, j) * (σ[j] - σ[i]) * dΩ.J[j] * 2 * 2π / length(dΩ.z) for j in (iseven(i) ? (1:2:length(dΩ.z)) : (2:2:length(dΩ.z)))])

    dĉdnₓ = [oddeven1(i) + oddeven2(i) for i in 1:length(dΩ.z)]

    # add normal derivative of particular term
    return dĉdnₓ .- sum([-sqrt(s)/2π * besselk.(1, sqrt(s) .* abs.(dΩ.z .- x₀)) .* real(dot.(dΩ.z .- x₀, dΩ.n̂)) ./ abs.(dΩ.z .- x₀) for x₀ in x₀])

end


#############################
### SMALL TRAPPING REGION ###
#############################

nbins = 40
binfac = 5

# setup
mf = MatFile("~/Transport/diffusion/1PoreData2.mat")
totalTimes = get_variable(mf, "totalTimes")
p = sum(totalTimes .< 1e10) / length(totalTimes)
totalTimes = totalTimes[totalTimes .< 1e10]

τ = range( log10(1 + minimum(totalTimes)), log10(1 + maximum(totalTimes)), length = nbins * binfac )
T = 10 .^ τ .- 1

histogram(log10.(1 .+ totalTimes), bins = range( log10(1 + minimum(totalTimes)), log10(1 + maximum(totalTimes)), length = nbins), normalize = :probability, label=:none)

x₀ = [5.0 + im * 0.0] # point sources
N = 128; θ = [2π/N * k for k in 0:N-1] # discretization
eps = 2.0; dΩ = Boundary([eps * exp.(im * θ)]; bc=:pointsource)

# talbot contour
Nᵧ = 32; γ = TalbotContour(
    # z(θ; t)
    (θ; t=nothing) -> length(θ) / 2t * (-0.6122 .+ 0.5017 * θ .* cot.(0.6407 * θ) + 0.6245 * im .* θ),
    # z′(θ; t)
    (θ; t=nothing) -> length(θ) / 2t * (-0.32143919 * θ .* csc.(0.6407*θ).^2 + 0.5017 * cot.(0.6407*θ) .+ 0.6245*im)
)

# plot total flux
dcdnₓ = talbot_midpoint(s -> pointwise_flux(dΩ; s, x₀), γ, Nᵧ)
flux = [abs.(sum(dcdnₓ(t) .* dΩ.J)) * 2π / dΩ.N for t in T]


a = cumsum(prepend!([0.5*(flux[i+1] + flux[i]) * (T[i+1] - T[i]) for i in 1:length(T)-1], 0.0))
plot!(τ[2:end], diff(a) * binfac / p, label=:none)

# plot(log10.(1 .+ T[2:end]), diff(a) * binfac / p)

# plot(T, flux, label=:none, xaxis=:log)
# xaxis!("Time"); yaxis!("Flux")

# plot analytic flux
function flux_exact(t)
    return mat"""
        R = norm($x₀) / $eps;
        fun = @(w,t) (2/pi).*((bessely(0,R.*w).*besselj(0,w) - bessely(0,w).*besselj(0,R.*w))./(besselj(0,w).^2+bessely(0,w).^2)).*w.*exp(-t.*w.^2);
        rho = @(t) integral(@(w) fun(w,t),0,Inf, 'AbsTol',1e-12,'RelTol',1e-12);
        rho($t / $eps / $eps);
    """
end
plot(T, flux_exact.(T) / eps^2, label=:none)



###################
### THREE PORES ###
###################

nbins = 40
binfac = 5

# setup
mf = MatFile("~/Transport/diffusion/3PoreData.mat")
totalTimes = get_variable(mf, "totalTimes")
p = sum(totalTimes .< 1e10) / length(totalTimes)
totalTimes = totalTimes[totalTimes .< 1e10]

τ = range( log10(1 + minimum(totalTimes)), log10(1 + maximum(totalTimes)), length = nbins * binfac )
T = 10 .^ τ .- 1

histogram(log10.(1 .+ totalTimes), bins = range( log10(1 + minimum(totalTimes)), log10(1 + maximum(totalTimes)), length = nbins), normalize = :probability, label=:none)

x₀ = [0.0 + im * 0.0] # point sources
N = 64; θ = [2π/N * k for k in 0:N-1] # discretization
dΩ = Boundary([
    1.0/8 * exp.(im * θ) .+ (3.0 + 3.0im),
    1.0/6 * exp.(im * θ) .+ (8.0 + 8.0im),
    1.0/3 * exp.(im * θ) .+ (10.0 + 10.0im)];
    bc=:pointsource)

# talbot contour
Nᵧ = 32; γ = TalbotContour(
    # z(θ; t)
    (θ; t=nothing) -> length(θ) / 2t * (-0.6122 .+ 0.5017 * θ .* cot.(0.6407 * θ) + 0.6245 * im .* θ),
    # z′(θ; t)
    (θ; t=nothing) -> length(θ) / 2t * (-0.32143919 * θ .* csc.(0.6407*θ).^2 + 0.5017 * cot.(0.6407*θ) .+ 0.6245*im)
)

# plot total flux
dcdnₓ = talbot_midpoint(s -> pointwise_flux(dΩ; s, x₀), γ, Nᵧ)
flux = [abs.(sum(dcdnₓ(t) .* dΩ.J)) * 2π / dΩ.N for t in T]


a = cumsum(prepend!([0.5*(flux[i+1] + flux[i]) * (T[i+1] - T[i]) for i in 1:length(T)-1], 0.0))
plot!(τ[2:end], diff(a) * binfac / p * dΩ.M, label=:none)

#################
### SIX PORES ###
#################

nbins = 40
binfac = 5

# setup
mf = MatFile("~/Transport/diffusion/6PoreData.mat")
totalTimes = get_variable(mf, "totalTimes")
p = sum(totalTimes .< 1e3) / length(totalTimes)
totalTimes = totalTimes[totalTimes .< 1e3]

τ = range( log10(1 + minimum(totalTimes)), log10(1 + maximum(totalTimes)), length = nbins * binfac )
T = 10 .^ τ .- 1

histogram(log10.(1 .+ totalTimes), bins = range( log10(1 + minimum(totalTimes)), log10(1 + maximum(totalTimes)), length = nbins), normalize = :probability, label=:none)

x₀ = [0.0 + im * 0.0] # point sources
N = 64; θ = [2π/N * k for k in 0:N-1] # discretization
dΩ = Boundary([
    0.275 * exp.(im * θ) .+ (-3.0 + 0.0im),
    0.02 * exp.(im * θ) .+ (0.0 + -2.0im),
    0.02 * exp.(im * θ) .+ (sqrt(2.0) + -sqrt(2.0)im),
    0.02 * exp.(im * θ) .+ (2.0 + 0.0im),
    0.02 * exp.(im * θ) .+ (sqrt(2.0) + sqrt(2.0)im),
    0.02 * exp.(im * θ) .+ (0.0 + 2.0im)];
    bc=:pointsource)

# talbot contour
Nᵧ = 32; γ = TalbotContour(
    # z(θ; t)
    (θ; t=nothing) -> length(θ) / 2t * (-0.6122 .+ 0.5017 * θ .* cot.(0.6407 * θ) + 0.6245 * im .* θ),
    # z′(θ; t)
    (θ; t=nothing) -> length(θ) / 2t * (-0.32143919 * θ .* csc.(0.6407*θ).^2 + 0.5017 * cot.(0.6407*θ) .+ 0.6245*im)
)

# plot total flux
dcdnₓ = talbot_midpoint(s -> pointwise_flux(dΩ; s, x₀), γ, Nᵧ)
flux = [abs.(sum(dcdnₓ(t) .* dΩ.J)) * 2π / dΩ.N for t in T]


a = cumsum(prepend!([0.5*(flux[i+1] + flux[i]) * (T[i+1] - T[i]) for i in 1:length(T)-1], 0.0))
plot!(τ[2:end], diff(a) * binfac / p * dΩ.M, label=:none)
