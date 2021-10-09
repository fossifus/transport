# NORMAL DERIVATIVE ON BOUNDARY

using LinearAlgebra
using IterativeSolvers: gmres
using SpecialFunctions
using Plots

includet("Talbot.jl")
includet("Yukawa.jl")

using .Yukawa
using .Talbot

f(x, s) = -1.0/s

function normal_derivative(dΩ, f; s)
    n̂, J, κ = Yukawa.GeomDerivs(dΩ);
    σ, gmres_log = gmres(I/2 + Yukawa.DLP(dΩ, s), f.(dΩ, s); reltol = 1e-12, maxiter=128, log=true);
    
    x = [real(dΩ) imag(dΩ)]
    K1(i, j) = besselk(1, sqrt(s) * norm(x[i,:] - x[j,:]))
    K2(i, j) = norm(x[i,:] - x[j,:])
    K3(i, j) = dot(x[i,:] - x[j,:], [real(n̂[j]) imag(n̂[j])])
    dK1(i, j) = sqrt(s) * (-besselk(0, sqrt(s) * norm(x[i,:] - x[j,:])) - besselk(1, sqrt(s) * norm(x[i,:] - x[j,:])) / norm(x[i,:] - x[j,:]) / sqrt(s)) * (x[i,:] - x[j,:])' / norm(x[i,:] - x[j,:])
    dK2(i, j) = (x[i,:] - x[j,:])' / norm(x[i,:] - x[j,:])
    dK3(i, j) = [real(n̂[j]) imag(n̂[j])]
    K(i, j) = sqrt(s) / 2π * dot([real(n̂[i]) imag(n̂[i])], (dK1(i, j) * K2(i, j) - K1(i, j) * dK2(i, j)) * K3(i, j) / K2(i, j) / K2(i, j) + K1(i, j) * dK3(i, j) / K2(i, j))

    # (K - λ) term
    dxdylog(i, j) = -real(dot(n̂[i], n̂[j])) / abs(dΩ[i] - dΩ[j])^2 + 2(real(dot(dΩ[i] - dΩ[j], n̂[i])) * real(dot(dΩ[i] - dΩ[j], n̂[j]))) / abs(dΩ[i] - dΩ[j])^4
    λ = -1 / 2π
    oddeven1(i) = sum([(K(i,j) - λ * dxdylog(i, j)) * σ[j] * J[j] * 2 * 2π / length(dΩ) for j in (iseven(i) ? (1:2:length(dΩ)) : (2:2:length(dΩ)))])

    # + λ term
    oddeven2(i) = sum([λ * dxdylog(i, j) * (σ[j] - σ[i]) * J[j] * 2 * 2π / length(dΩ) for j in (iseven(i) ? (1:2:length(dΩ)) : (2:2:length(dΩ)))])

    dĉdnₓ = [oddeven1(i) + oddeven2(i) for i in 1:length(dΩ)]

    # add normal derivative of particular term
    return dĉdnₓ

end

# invert laplace transform
Nᵧ = 32; γ = TalbotContour(
    # z(θ; t)
    (θ; t=nothing) -> length(θ) / 2t * (-0.6122 .+ 0.5017 * θ .* cot.(0.6407 * θ) + 0.6245 * im .* θ),
    # z′(θ; t)
    (θ; t=nothing) -> length(θ) / 2t * (-0.32143919 * θ .* csc.(0.6407*θ).^2 + 0.5017 * cot.(0.6407*θ) .+ 0.6245*im)
)
dcdnₓ = talbot_midpoint(s -> normal_derivative(dΩ, f; s), γ, Nᵧ)

#################
### UNIT BALL ###
#################

N = 128; θ = [2π/N * k for k in 1:N]
dΩ = exp.(im*θ)

### PLOT TOTAL FLUX 
T = range(0.1; step=0.5, stop=7.0)
dcdnₓ = talbot_midpoint(s -> normal_derivative(dΩ, f; s), γ, Nᵧ)
plt = plot(T, [abs.(sum(dcdnₓ(t))) * 2π / N for t in T], label=:none)
xaxis!("Time"); yaxis!("Flux")
savefig(plt, "flux.png")

### PLOT POINTWISE FLUX
T = [0.1, 0.5, 1.0, 5.0, 10.0]
plt = plot()
for t in T
    plot!(θ, abs.(dcdnₓ(t)), label="t=$t")
end
display(plt)
savefig(plt, "pointwise.png")

### CONVERGENCE
# exact flux
function flux_exact(t; m = 20)
    μ = approx_besselroots(0, m)
    cᵢ = (besselj.(1, μ) ./ μ) ./ (0.5 * besselj.(1, μ) .^ 2)
    return abs(2π * sum(cᵢ .* exp.(-μ.^2 * t) .* -μ .* besselj.(1, μ)))
end
# plot convergence
t_test = 0.32; err = []
for N in [64, 128, 256]#, 512, 1024]
    @info N
    θ = [2π/N * k for k in 1:N]; dΩ = exp.(im*θ)
    dcdnₓ = talbot_midpoint(s -> normal_derivative(dΩ, f; s), γ, Nᵧ)
    fl_ex = flux_exact(t_test);
    @time fl_ap = abs(sum(dcdnₓ(t_test))) * 2π / N
    push!(err, abs(fl_ex - fl_ap) / abs(fl_ex))
end
plot([64, 128, 256], err, xaxis=:log, yaxis=:log, marker=:square, label=:none, title="Error in Flux, Unit Ball")
xlabel!("# Boundary Points")
ylabel!("Error")
savefig("convergence.png")