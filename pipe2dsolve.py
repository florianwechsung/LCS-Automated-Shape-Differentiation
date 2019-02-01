from firedrake import *

mesh = Mesh("pipe.msh")

degree = 2
V = VectorFunctionSpace(mesh, "CG", degree)
Q = FunctionSpace(mesh, "CG", degree-1)
Z = V * Q

z = Function(Z)
test = TestFunction(Z)

nu = 1./400.
u, p = split(z)
v, q = split(test)
e = nu*inner(grad(u), grad(v))*dx - p*div(v)*dx + inner(dot(grad(u), u), v)*dx + div(u)*q*dx


X = SpatialCoordinate(mesh)
uin = 6 * as_vector([(1-X[1])*X[1], 0])
bcs = [DirichletBC(Z.sub(0), Constant((0, 0)), [3, 4]), DirichletBC(Z.sub(0), uin, 1)]


sp = {
    "mat_type": "aij",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}

solve(e==0, z, bcs=bcs, solver_parameters=sp)

out = File("pipe2dns.pvd")
out.write(z.split()[0])

