#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdlib.h>
#include <stdio.h>

#include <mpi.h>

#include <vector>
#include <utility>
#include <limits>
#include <sstream>

namespace {
	char message[MPI_MAX_ERROR_STRING];

	static void handleError(int err, const char *file, int line) {
		if (err != MPI_SUCCESS) {
			int length;
			MPI_Error_string(err, message, &length);
			fprintf(stderr, "ERROR IN: %s:%d: %.*s\n", file, line, length, message);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}
}

#define HE( err) (handleError( err, __FILE__, __LINE__ ))

using seed_t = std::pair<int, int>;

struct PArgs
{
	size_t n;
	size_t m;
	size_t k;
	std::vector<seed_t> seeds;
	bool v_flag;
	bool g_flag;
	double ge_value;
};

PArgs parse_args(int argc, char **argv)
{
	size_t n, m, k;
	std::vector<seed_t> seeds;
	bool v_flag = false;
	bool g_flag = false;
	double ge_value = std::numeric_limits<double>::quiet_NaN();

	sscanf(argv[1], "%zu", &n);
	sscanf(argv[2], "%zu", &m);
	sscanf(argv[3], "%zu", &k);

	for (int i = 4; i < argc; i++)
	{
		if (argv[i][1] == 'v')
		{
			v_flag = true;
		}
		else if (argv[i][1] == 'g')
		{
			g_flag = true;
			sscanf(argv[i + 1], "%lf", &ge_value);
			i++;
		}
		else if (argv[i][1] == 's')
		{
			std::stringstream ss(argv[i + 1]);
			int seed1, seed2;
			while (ss.good())
			{
				ss >> seed1;
				if (ss.peek() == ',')
            		ss.ignore();
				if (ss.good())
				{
					ss >> seed2;
					if (ss.peek() == ',')
            			ss.ignore();
					seeds.push_back(seed_t(seed1, seed2));
				}
			}
		}
	}

	PArgs pargs = {n, m, k, seeds, v_flag, g_flag, ge_value};
	return pargs;
}



#endif /* __UTILS_H__ */