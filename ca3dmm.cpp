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
	// if(myRank == 0)
	// {
	// 	std::cout << "n: " << args.n << std::endl;
	// 	std::cout << "m: " << args.m << std::endl;
	// 	std::cout << "k: " << args.k << std::endl;
	// 	std::cout << "v: " << args.v_flag << std::endl;
	// 	std::cout << "g: " << args.g_flag << std::endl;
	// 	std::cout << "ge: " << args.ge_value << std::endl;
	// 	std::cout << "seeds: " << std::endl;
	// 	for (auto seed : args.seeds) {
	// 		std::cout << seed.first << " " << seed.second << std::endl;
	// 	}
	// }

	int pm, pn, pk;
	std::tie(pm, pn, pk) = solve_for_pmpnpk(args.n, args.m, args.k, numProcesses);
	// if(myRank == 0) {

	// 	std::cout << "pm: " << pm << std::endl;
	// 	std::cout << "pn: " << pn << std::endl;
	// 	std::cout << "pk: " << pk << std::endl;
	// }

	// Exit processes with ranks not in [0, pm * pn * pk - 1]
	MPI_Group world_group;
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);
	MPI_Group new_group;
	int ranges[1][3] = {{ pm * pn * pk, numProcesses-1, 1 }};
	MPI_Group_range_excl(world_group, 1, ranges, &new_group);
	MPI_Comm newworld;
	MPI_Comm_create(MPI_COMM_WORLD, new_group, &newworld);

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

	MPI_Finalize();
	return 0;
}