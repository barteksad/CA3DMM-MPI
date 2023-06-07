#include <mpi.h>

#include <iostream>
#include <string>

#include "ca3dmm.h"
#include "utils.h"

int main(int argc, char **argv) {
	int numProcesses;
	int myRank;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

	PArgs args = parse_args(argc, argv);

	int pm, pn, pk;
	std::tie(pm, pn, pk) = solve_for_pmpnpk(args.n, args.m, args.k, numProcesses);

	// Exit processes with ranks not in [0, pm * pn * pk - 1]
	MPI_Comm newworld;
	if(pm * pn * pk == numProcesses) {
		newworld = MPI_COMM_WORLD;
	} else {
		MPI_Group world_group;
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);
		MPI_Group new_group;
		int ranges[1][3] = {{ pm * pn * pk, numProcesses-1, 1 }};
		MPI_Group_range_excl(world_group, 1, ranges, &new_group);
		MPI_Comm_create(MPI_COMM_WORLD, new_group, &newworld);
	}

	if (newworld == MPI_COMM_NULL)
	{
		MPI_Finalize();
		exit(0);
	}
	else 
	{
		MatrixProcess mp(args, pm, pn, pk, myRank, newworld);
		mp.run();
	}

	HE(MPI_Finalize());
	return 0;
}