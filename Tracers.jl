# params = ParamSet(
# 		# The data file for the initial geometry
# 		infolder = "advection/input_geos/ten_bodies/",
# 		label = "10circ42",
# 		# Varied computational parameters
# 		npts = 128,                # The number of points per body, default 128.
# 		ibary = 1,                        # Use barycentric (1) or not (0).
# 		ifmm = 1,                        # Use the FFM (1) or not (0).
# 		ibc = 0,                        # Use slip BCs (1) or no-slip (0)
# 		# Varied physical parameters
# 		epsfac = 15,        # Smoothing parameter for curvature driven flow.
# 		sigfac = 10,        # Smoothing parameter for the stress.
# 		dt = 1e-2,                # The time step.
# 		outstride = 1,                # The number of steps per output, default 4.
# 		# Fixed physical parameters
# 		fixpdrop = 1,                # Fix the pressure drop (1) or not (0)
# 		fixarea = 0,                # Keep the area fixed (1) or not (0)
# 		tfin = 1e-2,                # The final time.
# 		# Fixed computational parameters
# 		maxl = 5000,                # The maximum number of GMRES iterations.
# 		nouter = 1024,                # The number of points on the outer boundary, default 1024.
# )
# run_erosion(params)

tracers_per_core = 10
min_tracer_length = 3.0
checkfreq = 100; checkthresh = 0.01
h = 0.01

# using Erosion
using Erosion.DensityStress
using Erosion.ThetaLen
using Distributions
using JLD2
using Parameters: @unpack
using GeometricalPredicates: inpolygon, Polygon, Point

thlenden_noutput = 2

jf = load("raw_data-10circ42.jld2")
params = jf["params"]
thlenden = jf["thldvec"][thlenden_noutput]

# choose start point
p = rand(Uniform(-1.0, 1.0))

# solve for density kernel
compute_density!(thlenden, params)

# get params
@unpack npts, nouter, ibary = params
nbods, xv, yv, dens = getnxyden(thlenden, params, false, false)

# get bodies
xx = reshape(xv, (params.npts,length(xv) ÷ params.npts))
yy = reshape(yv, (params.npts,length(yv) ÷ params.npts))
bodies = [Polygon(Point.(xx,yy)...) for (xx,yy) in zip([xx[:,i] for i in 1:size(xx)[2]], [yy[:,i] for i in 1:size(yy)[2]])]

# precompute velocities on grid of possible start points
start_x = range(-0.25, -0.5; length=20); start_y = range(-0.75, 0.75; length=40)
start_grid = TargetsType([x for x in start_x for y in start_y], [y for x in start_x for y in start_y], [], [], [], [])
compute_qoi_targets!(thlenden, start_grid, params)

# build tracers
tracers = []
while length(tracers) < tracers_per_core

    # choose random start location
    posx = [start_x[rand(1:length(start_x))]]
    posy = [start_y[rand(1:length(start_y))]]
    path = [(posx[1], posy[1])]
    len = 0.0; len100 = 0.0; it = 0

    # enforce minimum tracer length
    while len < min_tracer_length

        # count
        @info it += 1; @show len

        # do RK4 step
        @time u1, v1, ptar, vortar = compute_qoi_targets(xv,yv,dens,
            posx,posy,
            npts,nbods,nouter,ibary)
        u1 = u1[1]; v1 = v1[1]

        # abort if in body or stuck
        if norm([u1, v1]) < checkthresh
            @info "BREAK"
            break
        end

        u2, v2, ptar, vortar = compute_qoi_targets(xv,yv,dens,
            posx .+ h/2 * u1, posy .+ h/2 * v1,
            npts,nbods,nouter,ibary)
        u2 = u2[1]; v2 = v2[1]

        u3, v3, ptar, vortar = compute_qoi_targets(xv,yv,dens,
            posx .+ h/2 * u2, posy .+ h/2 * v2,
            npts,nbods,nouter,ibary)
        u3 = u3[1]; v3 = v3[1]

        u4, v4, ptar, vortar = compute_qoi_targets(xv,yv,dens,
            posx .+ h * u3, posy .+ h * v3,
            npts,nbods,nouter,ibary)
        u4 = u4[1]; v4 = v4[1]

        # new pos
        posx .+= h/6 * (u1 + 2u2 + 2u3 + u4)
        posy .+= h/6 * (v1 + 2v2 + 2v3 + v4)

        # update path
        len += norm([path[end][1] - posx[1], path[end][2] - posy[1]])
        push!(path, (posx[1], posy[1]))

        # abort if in body
        # if xor([inpolygon(bod, Point(posx[1], posy[1])) for bod in bodies]...)
        #     break
        # end

        # reinsert at nearest velocity if out of bounds
        if posx[1] > 0.99 || posx[1] < -0.99 || posy[1] > 0.99 || posy[1] < -0.99
            @info "REINSERTION"
            index = findmin(norm.(u2 .- start_grid.utar, v2 .- start_grid.vtar))[2]
            posx = [start_grid.xtar[index]]; posy = [start_grid.ytar[index]]
            push!(path, (posx[1], posy[1]))
        end

        # abort if too slow
        # if it % checkfreq == 0
        #     if len - len100 < checkthresh
        #         break
        #     end
        #     len100 = len
        # end
        
        # save tracer
        if len >= min_tracer_length
            push!(tracers, path)
        end

    end

end