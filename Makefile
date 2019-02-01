all:
	gmsh -2 -clscale 0.05 pipe2d.geo
	gmsh -3 -clscale 0.1 pipe3d.geo
