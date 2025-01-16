# FEM za 1D Poissonov problem
#   -u'' = x*(1-x) na intervalu (0,1)
#   u(0) = u(1) = 0
#   egzaktno rješenje u = x^4/12 - x^3/6 + x/12

from dolfinx import mesh, fem, io, plot
import ufl
from petsc4py.PETSc import ScalarType

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

mpi_comm = MPI.COMM_WORLD

# Domena i funkcijski prostor
#---------------------------------------------------------
nx = 8
domena = mesh.create_unit_interval(mpi_comm, nx)
x = ufl.SpatialCoordinate(domena)
V = fem.FunctionSpace(domena, ('CG', 1))

# Rub domene i rubni uvjeti
#---------------------------------------------------------
dim_domena = domena.topology.dim
dim_rub = dim_domena - 1
domena.topology.create_connectivity(dim_rub, dim_domena)
rub_indeks = mesh.exterior_facet_indices(domena.topology)
rub_dofs = fem.locate_dofs_topological(V, dim_rub, rub_indeks)

uD = fem.Constant(domena, ScalarType(0))
ru = fem.dirichletbc(uD, rub_dofs, V)

# Slaba formulacija
#---------------------------------------------------------
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

f = fem.Function(V)
f.interpolate(lambda x: x[0]*(1 - x[0]))

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
b = f * v * ufl.dx

# Solver
#---------------------------------------------------------
problem = fem.petsc.LinearProblem(a, b, bcs=[ru])
uh = problem.solve()


# Stvarno rješenje
#---------------------------------------------------------
n_egz = 1000
domena_egz = mesh.create_unit_interval(mpi_comm, n_egz)
V_egz = fem.FunctionSpace(domena_egz, ('CG', 1))
u_egz = fem.Function(V_egz)
u_egz.interpolate(lambda x: np.power(x[0],4)/12 - np.power(x[0],3)/6 + x[0]/12)

u = fem.Function(V_egz)
u.interpolate(uh)

L2_norma = fem.form(ufl.inner(u - u_egz, u - u_egz) * ufl.dx)
L2_greska_lokalna = fem.assemble_scalar(L2_norma)
L2_greska = np.sqrt(domena_egz.comm.allreduce(L2_greska_lokalna, op=MPI.SUM))

print(L2_greska)

# Spremanje rješenja
#---------------------------------------------------------
with io.XDMFFile(domena.comm, "poisson1D.xdmf", "w") as xdmf:
    xdmf.write_mesh(domena)
    xdmf.write_function(uh)

with io.XDMFFile(domena.comm, "poisson1D_egz.xdmf", "w") as xdmf:
    xdmf.write_mesh(domena)
    xdmf.write_function(u_egz)

# Graf
#---------------------------------------------------------
cells, types, x = plot.create_vtk_mesh(V)
cells_egz, types_egz, x_egz = plot.create_vtk_mesh(V_egz)

plt.rcParams.update({'font.size': 13})
fig, ax = plt.subplots(figsize=(10, 8))
plt.plot(x_egz[:,0], u_egz.x.array.real, linewidth=2, color = '#4285F4', label = 'egzaktno rješenje')
plt.plot(x[:,0], uh.x.array.real, linewidth=2, color = '#EA4335', label = 'numeričko rješenje')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_title('Usporedba numeričkog i egzaktnog rješenja')
plt.grid()
plt.savefig('poisson1D.png', bbox_inches='tight')
plt.show()
plt.close()
