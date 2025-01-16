# Konvergencija FEM-a za 1D Poissonov problem
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

def poisson1D_solver(nx):

    domain = mesh.create_unit_interval(mpi_comm, nx)
    V = fem.FunctionSpace(domain, ('CG', 1))

    dim_domena = domain.topology.dim
    dim_rub = dim_domena - 1
    domain.topology.create_connectivity(dim_rub, dim_domena)
    rub_indeks = mesh.exterior_facet_indices(domain.topology)
    rub_dofs = fem.locate_dofs_topological(V, dim_rub, rub_indeks)

    uD = fem.Constant(domain, ScalarType(0))
    ru = fem.dirichletbc(uD, rub_dofs, V)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    f = fem.Function(V)
    f.interpolate(lambda x: x[0]*(1 - x[0]))
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    b = f * v * ufl.dx

    problem = fem.petsc.LinearProblem(a, b, bcs=[ru])
    uh = problem.solve()

    return uh


# Egzaktno rješenje
#---------------------------------------------------------
n_egz = 10000
domain_egz = mesh.create_unit_interval(mpi_comm, n_egz)
V_egz = fem.FunctionSpace(domain_egz, ('CG', 1))
u_egz = fem.Function(V_egz)
u_egz.interpolate(lambda x: np.power(x[0],4)/12 - np.power(x[0],3)/6 + x[0]/12)

n_mjerenja = 20
h = np.zeros(n_mjerenja)
error_L2 = np.zeros(n_mjerenja)
error_H10 = np.zeros(n_mjerenja)

it = 0
for nx in range (10, 2000, 100):

    h[it] = 1/(nx-1)
    uh = poisson1D_solver(nx)

    u = fem.Function(V_egz)
    u.interpolate(uh)

    L2_error = fem.form(ufl.inner(u - u_egz, u - u_egz) * ufl.dx)
    error_local_L2 = fem.assemble_scalar(L2_error)
    error_L2[it] = np.sqrt(domain_egz.comm.allreduce(error_local_L2, op=MPI.SUM))

    H10_error = fem.form(ufl.inner(ufl.grad(u) - ufl.grad(u_egz), ufl.grad(u) - ufl.grad(u_egz)) * ufl.dx)
    error_local_H10 = fem.assemble_scalar(H10_error)
    error_H10[it] = np.sqrt(domain_egz.comm.allreduce(error_local_H10, op=MPI.SUM))
    it+=1

plt.rcParams.update({'font.size': 13})
fig, ax = plt.subplots(figsize=(10, 8))
plt.loglog(h, error_L2, linewidth=2, color = '#4285F4', label = 'L2 norma greške')
plt.loglog(h, error_H10, linewidth=2, color = '#EA4335', label = 'H10 norma greške')
ax.legend()
ax.set_xlabel('h')
ax.set_ylabel('greška')
ax.set_title('Konvergencija metode')
plt.grid()
plt.savefig('poisson1D_konvergencija.png', bbox_inches='tight')
plt.show()
plt.close()
