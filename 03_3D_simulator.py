#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 23:31:26 2025

@author: tommycursonsmith
"""

import numpy as np
from dedalus import public as d3
import logging
logger = logging.getLogger(__name__)

Lx, Ly, Lz = 8, 8, 1
Nx, Ny, Nz = 64, 64, 64
Rayleigh = 1.5e6
Prandtl = 1

dealias = 3/2
stop_sim_time = 40
max_timestep = 0.125
timestepper = d3.RK222
dtype = np.float64

coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

p = dist.Field(name='p', bases=(xbasis, ybasis, zbasis))
b = dist.Field(name='b', bases=(xbasis, ybasis, zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis, ybasis, zbasis))

tau_p = dist.Field(name='tau_p')
tau_b1 = dist.Field(name='tau_b1', bases=(xbasis, ybasis))
tau_b2 = dist.Field(name='tau_b2', bases=(xbasis, ybasis))
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=(xbasis, ybasis))
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=(xbasis, ybasis))

kappa = (Rayleigh * Prandtl)**(-0.5)
nu = (Rayleigh / Prandtl)**(-0.5)
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
ex, ey, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez * lift(tau_u1)
grad_b = d3.grad(b) + ez * lift(tau_b1)

problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")

problem.add_equation("b(z=0) = Lz")
problem.add_equation("b(z=Lz) = 0")
problem.add_equation("u(z=0) = 0")
problem.add_equation("u(z=Lz) = 0")
problem.add_equation("integ(p) = 0")

solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

b.fill_random('g', seed=42, distribution='normal', scale=1e-3)
b['g'] *= z * (Lz - z)
b['g'] += Lz - z  

snapshots = solver.evaluator.add_file_handler('snapshots_3D', sim_dt=(stop_sim_time/3)-0.2, max_writes=1)
snapshots.add_task(b, name='buoyancy')
snapshots.add_task((d3.curl(u)@d3.curl(u))**0.5, name='vorticity')

CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name='Re')

startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration - 1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info(f"Iteration={solver.iteration}, Time={solver.sim_time:.4f}, dt={timestep:.4f}, max(Re)={max_Re:.2f}")
except Exception as e:
    logger.error('Exception raised, triggering end of main loop.')
    raise e
finally:
    solver.log_stats()
