# FEM za 2D Poissonov problem
#   -(u_xx + u_yy) = x*(1-x)*y*(2-y) na pravokutniku (0,1)*(0,2)
#   u = 0 na rubu
#   egzaktno rješenje u = x^4/12 - x^3/6 + x/12

from dolfinx import mesh, fem, io
import ufl
from petsc4py.PETSc import ScalarType

from mpi4py import MPI
import numpy as np

mpi_comm = MPI.COMM_WORLD

# Domena i funkcijski prostor
#---------------------------------------------------------
nx, ny = 20, 30
domain = mesh.create_rectangle(mpi_comm, [np.array([0, 0]), np.array([1, 2])], [nx, ny], mesh.CellType.triangle)
x = ufl.SpatialCoordinate(domain)
V = fem.FunctionSpace(domain, ('CG', 1))

# Rub domene i rubni uvjeti
#---------------------------------------------------------
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

uD = fem.Constant(domain, ScalarType(0))
bc = fem.dirichletbc(uD, boundary_dofs, V)

# Slaba formulacija
#---------------------------------------------------------
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

f = fem.Function(V)
f.interpolate(lambda x: 10*x[0]*(1 - x[0])*x[1]*(2 - x[1]))

a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# Spremanje rješenja
#---------------------------------------------------------
with io.XDMFFile(domain.comm, "poisson2D.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)
