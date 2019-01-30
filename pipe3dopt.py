from firedrake import *

# mesh = Mesh("pipe.msh")
mesh = Mesh("3dpipe.msh")
coords = mesh.coordinates.vector()
X = SpatialCoordinate(mesh)

W = mesh.coordinates.function_space()
gradJ = Function(W)
phi, psi = TrialFunction(W), TestFunction(W)
A_riesz = assemble(inner(grad(phi), grad(psi)) * dx)

Z = VectorFunctionSpace(mesh, "CG", 2) \
    * FunctionSpace(mesh, "CG", 1)
z, z_adjoint = Function(Z), Function(Z)
u, p = split(z)
test = TestFunction(Z)
v, q = split(test)

nu = Constant(1./100.)
e = nu*inner(grad(u), grad(v))*dx - p*div(v)*dx \
    + inner(dot(grad(u), u), v)*dx + div(u)*q*dx
uin = 6 * as_vector([(1-X[1])*X[1], 0])
r = sqrt(X[0]**2+X[1]**2)
uin = as_vector([0, 0, 1-(2*r)**2])
bcs = [DirichletBC(Z.sub(0), 0., [3, 4]),
       DirichletBC(Z.sub(0), uin, 1)]
sp = {"pc_type": "lu", "mat_type": "aij",
      "pc_factor_mat_solver_type": "mumps", "ksp_monitor": None}

J = nu * inner(grad(u), grad(u)) * dx
volume = Constant(1.) * dx(domain=mesh)
target_volume = assemble(volume)
dvol = derivative(volume, X)
c = 0.1
L = replace(e, {test: z_adjoint}) + J
dL = derivative(L, X)

out = File("u.pvd")
def solve_state_and_adjoint():
    solve(e==0, z, bcs=bcs, solver_parameters=sp)
    solve(derivative(L, z)==0, z_adjoint,
          bcs=homogenize(bcs), solver_parameters=sp)
    out.write(z.split()[0])

solve_state_and_adjoint()
nu.assign(1./400)
solve_state_and_adjoint()
nu.assign(1./1000)
solve_state_and_adjoint()
for i in range(100):
    dJ = assemble(dL).vector() \
        + assemble(dvol).vector() * c * 2\
        * (assemble(volume)-target_volume)
    solve(A_riesz, gradJ, dJ,
          bcs=DirichletBC(W, 0, [1, 2, 3]))
    print("i = %3i; J = %.6f; ||dJ|| = %.6f"
          % (i, assemble(J), norm(grad(gradJ))))
    coords -= 0.3 * gradJ.vector()
    solve_state_and_adjoint()

