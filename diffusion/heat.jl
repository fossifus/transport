# using Plots
using SpecialFunctions
using FFTW
using LinearAlgebra
using IterativeSolvers
using MATLAB
using FastGaussQuadrature
using HypergeometricFunctions
using GeometricalPredicates: inpolygon, Polygon, Point

includet("Talbot.jl")
includet("Fourier.jl")
includet("Yukawa.jl")

using .Talbot
using .Fourier
using .Yukawa

mutable struct Boundary
    dΩ::Array{Array{Complex{Float64}, 1}, 1}
    M::Integer # number of bodies
    N::Integer # discretization pts per body
    Boundary(dΩ::Array{Complex{Float64}, 1}) = new([dΩ], 1, length(dΩ[begin])) #simply connected
    Boundary(dΩ::Array{Array{Complex{Float64}, 1}, 1}) = new(dΩ, length(dΩ), length(dΩ[begin])) #multiply connected
end

### BOUNDARY DISCRETIZATION ###
# N = 128; θ = [2π/N * k for k in 1:N] # CCW, interior
N = 128; θ = reverse!([2π/N * k for k in 1:N]) # CW, exterior
# dΩ = Boundary(exp.(im*θ) .* (3 .+ sin.(5θ))) # star
dΩ = Boundary(exp.(im*θ)) # circle
# dΩ = Boundary([exp.(im*θ) .+ 2.0im, exp.(im*θ) .- 2.0im]) # 2 circ

### EVALUATION GRID ###
Nᵣ = 16; Nᵩ = 64; Rmin = 1.2; Rmax = 5;
r = range(Rmin; stop=Rmax, length=Nᵣ)
φ = [2π/(Nᵩ-1) * k for k in 1:Nᵩ]
Ω = exp.(im*φ) * r' # circle
# Ω = exp.(im*φ) .* (3 .+ sin.(5φ)) * r' # star

x = range(-2, 2; length=50)
y = im * range(-3.5, 3.5; length=100)
Ω = (x * ones(length(y))')' .+ y * ones(length(x))'

### PLOT GEOMETRY ###
# n̂, J, κ = Yukawa.GeomDerivs(dΩ)
# n̂ = vcat([d[1] for d in Yukawa.GeomDerivs.(dΩ)]...)
# J = vcat([d[2] for d in Yukawa.GeomDerivs.(dΩ)]...)
# κ = vcat([d[3] for d in Yukawa.GeomDerivs.(dΩ)]...)
# dΩ = vcat(dΩ...)
# plot(dΩ; aspect_ratio=1)
# quiver!(real(dΩ[1:10:end]), imag(dΩ[1:10:end]), quiver=(real(n̂[1:10:end]), imag(n̂[1:10:end])))
# quiver!(real(dΩ[1:10:end]), imag(dΩ[1:10:end]), quiver=(real(im*n̂[1:10:end]), imag(im*n̂[1:10:end])))
# scatter!(Ω, label=:none, color=:black, markersize=1)

### PLOT MULTI BOD ###
# plt = plot()
# for dΩᵢ in dΩ
#     n̂, J, κ = Yukawa.GeomDerivs(dΩᵢ)
#     plot!(dΩᵢ; aspect_ratio=1)
#     quiver!(real(dΩᵢ[1:10:end]), imag(dΩᵢ[1:10:end]), quiver=(real(n̂[1:10:end]), imag(n̂[1:10:end])))
#     quiver!(real(dΩᵢ[1:10:end]), imag(dΩᵢ[1:10:end]), quiver=(real(im*n̂[1:10:end]), imag(im*n̂[1:10:end])))
# end
# display(plt)
# scatter!(Ω, label=:none, color=:black, markersize=1)

### BOUNDARY CONDITION ###
f(x, s) = -1.0/s

# x₀ = [complex(5.0)]
# f(x, s) = -sum([1/2π * besselk(0, sqrt(s) * norm(x - x₀)) for x₀ in x₀])

### TALBOT CONTOUR ###
Nᵧ = 32; γ = TalbotContour(
    # z(θ; t)
    (θ; t=nothing) -> length(θ) / 2t * (-0.6122 .+ 0.5017 * θ .* cot.(0.6407 * θ) + 0.6245 * im .* θ),
    # z′(θ; t)
    (θ; t=nothing) -> length(θ) / 2t * (-0.32143919 * θ .* csc.(0.6407*θ).^2 + 0.5017 * cot.(0.6407*θ) .+ 0.6245*im)
)

function BIESolve(t)

    @info "Time $t"

    ### GET CONTOUR POINTS ###
    s = Talbot.γ(γ, Nᵧ, t); s′ = Talbot.γ′(γ, Nᵧ, t);
    
    ### SOLVE SKIE AT TALBOT CONTOUR POINTS ###
    # @time C = [YukawaSolveMatrixFree(dΩ, f, s) for s in s]
    @time C = [YukawaSolve(dΩ, f, s) for s in s]
    
    ### BROMWICH INTEGRAL ###
    c(x,t) = -im / Nᵧ * sum(exp.(s*t) .* [C(x,s) for (C,s) in zip(C,s)] .* s′)

    ### EVALUATE ON TARGET GRID Ω ###
    return @time c.(Ω, t)
    
end

### LOOP OVER TIME ###
# t = [0.1, 0.15, 0.2, 0.25]
# t = [0.2, 5.0, 10.0]
t = [1.0]
c = [BIESolve(t) for t in t]

### set concentration within body
bodies = [Polygon(Point.(xx,yy)...) for (xx,yy) in zip(real.(dΩ.dΩ), imag.(dΩ.dΩ))]
inside = findall(xor.([inpolygon.(Ref(bod), [Point(xx,yy) for (xx,yy) in zip(real.(Ω), imag.(Ω))]) for bod in bodies]...))
for cᵢ in c
    cᵢ[inside] .= 0
end

### PLOTTING ###
for i in 1:length(c)
    z = real(c[i]);
    mat"""
    i = $i;
    figure()
    surf(real($Ω), imag($Ω), $z)
    view(2)
    shading interp
    axis equal
    #axis([-1 1 -1 1])
    #caxis([0 1])
    colorbar
    exportgraphics(gcf,\"heat\" + i + \".png\",'Resolution',300)
    """
end

plt = plot()
for i in 1:length(c)
    plot!(real(Ω[1,:]), real(c[i][1,:]))
end
display(plt)

for i in 1:length(c)
    z = real(c[i]);
    mat"""
    i = $i;
    figure()
    plot(abs($Ω)(:,1), $z(:,1))
    view(2)
    shading interp
    axis equal
    #axis([-1 1 -1 1])
    caxis([0 1])
    colorbar
    exportgraphics(gcf,\"heat\" + i + \".png\",'Resolution',300)
    """
end

### EXACT SOLUTION ###
function heat(r, t; m = 20)
    μ = approx_besselroots(0, m)
    cᵢ = (besselj.(1, μ) ./ μ) ./ (0.5 * besselj.(1, μ) .^ 2)
    return sum(cᵢ .* exp.(-μ.^2 * t) .* besselj.(0, μ * r))
end

c_exact = [heat.(abs.(Ω), t) for t in t]
for i in 1:length(c_exact)
    z = real(c_exact[i]);
    mat"""
    i = $i;
    figure()
    surf(real($Ω), imag($Ω), $z)
    view(2)
    shading interp
    axis equal
    axis([-1 1 -1 1])
    caxis([0 1])
    colorbar
    exportgraphics(gcf,\"heat_exact\" + i + \".png\",'Resolution',300)
    """
end

c_err = [abs.(err) for err in c .- c_exact]
for i in 1:length(c_err)
    z = real(c_err[i]);
    mat"""
    i = $i;
    figure()
    surf(real($Ω), imag($Ω), $z)
    view(2)
    shading interp
    axis equal
    axis([-1 1 -1 1])
    colorbar
    exportgraphics(gcf,\"heat_err\" + i + \".png\",'Resolution',300)
    """
end



#########################
### CONVERGENCE STUDY ###
#########################

### EVALUATION GRID STOPS AT R = 0.8 ###
Nᵣ = 16; Nᵩ = 64; Rmin = 0.01; Rmax = 0.8;
r = range(Rmin; stop=Rmax, length=Nᵣ)
φ = [2π/(Nᵩ-1) * k for k in 1:Nᵩ]
Ω = exp.(im*φ) * r'

err = []; t = 1.0; Ns = [64, 128, 256, 512]
for N in Ns
    θ = [2π/N * k for k in 1:N] # CCW, interior
    dΩ = exp.(im*θ) # circle
    push!(err, norm(BIESolve(t) .- heat.(abs.(Ω), t)) / norm(heat.(abs.(Ω), t)))
end

plot(Ns, err, yaxis=:log, xaxis=:log, marker=:square, label=:none, title="LogLog Error, HeatEq Unit Circle")
xaxis!("Boundary points on dΩ")
yaxis!("Error")
savefig("heat_convergence.png")

[err[i]/err[i+1] for i in 1:length(err)-1]