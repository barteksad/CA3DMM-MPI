#ifndef __CA3DMM_H__
#define __CA3DMM_H__

#include <mpi.h>

#include <tuple>
#include <cmath>
#include <cassert>
#include <vector>
#include <iostream>

#include "utils.h"
#include "densematgen.h"

std::tuple<int, int, int> solve_for_pmpnpk(int n, int m, int k, int p)
{
	/*
		This function uses Lagrange Multipliers to solve the following problem:
		minimize pk * m * n + pm * n * k + pn * m * k
		subject to: 0.95 * p <= pm * pn * pk <= p
				and mod(max(pm, pn), min(pm, pk)) == 0
	*/

	double startRange = std::ceil((n * m * k) / p);
	double endRange = std::floor((n * m * k) / (0.95 * p));

	double bestVal = std::numeric_limits<double>::max();
	int bestPm = 0, bestPn = 0, bestPk = 0;

	for(double lambda = startRange; lambda <= endRange; lambda++)
	{
		double lambda_sqrt = std::cbrt(lambda);
		int pm = m / lambda_sqrt;
		int pn = n / lambda_sqrt;
		int pk = k / lambda_sqrt;

		if(pm*pn*pk == 0) {
			continue;
		}

		if(std::max(pm, pn) % std::min(pm, pn) == 0) {
			double val = pk * m * n + pm * n * k + pn * m * k;
			if(val < bestVal) {
				bestVal = val;
				bestPm = pm;
				bestPn = pn;
				bestPk = pk;
			}
		}
	}

	if(bestVal == std::numeric_limits<double>::max()) {
		std::cerr << "No solution found" << std::endl;
		exit(1);
	}

	return std::make_tuple(bestPm, bestPn, bestPk);
}

enum class MatrixType
{
	A=0,
	B=1,
	C,
};

class Fragment
{
public:
	Fragment() {}
	Fragment(MatrixType type, int rowOffset, int colOffset, int numRows, int numCols)
	: data(numRows * numCols, 0), recvBuffer(numRows * numCols, 0), type(type), rowOffset(rowOffset), colOffset(colOffset), numRows(numRows), numCols(numCols) {}

	void generate(int seed) {
		assert(type != MatrixType::C);
		if(type == MatrixType::A) 
		{
			for(auto row = 0; row < numRows; row++) {
				for(auto col = 0; col < numCols; col++) {
					data[row * numCols + col] = generate_double(seed, row + rowOffset, col + colOffset);
				}
			}
		}
		else
		{
			for(auto col = 0; col < numCols; col++) {
				for(auto row = 0; row < numRows; row++) {
					data[col * numRows + row] = generate_double(seed, row + rowOffset, col + colOffset);
				}
			}
		}
	}

	void mul(const Fragment& A, const Fragment& B) {
		assert(type == MatrixType::C);
		assert(A.type == MatrixType::A);
		assert(B.type == MatrixType::B);

		for(auto row = 0; row < numRows; row++) {
			for(auto col = 0; col < numCols; col++) {
				double sum = 0;
				for(auto idx = 0; idx < A.numCols; idx++) {
					sum += A.data[row * A.numCols + idx] * B.data[col * B.numCols + idx];
				}
				data[row * numCols + col] += sum;
			}
		}
		
	}

	void sendTo(int dest, MPI_Comm comm, MPI_Request *sendRequest) {
		HR(MPI_Isend(data.data(), data.size(), MPI_DOUBLE, dest, 0, comm, sendRequest));
	}

	void recvFrom(int src, MPI_Comm comm, MPI_Request *recvRequest) {
		HR(MPI_Irecv(recvBuffer.data(), recvBuffer.size(), MPI_DOUBLE, src, 0, comm, recvRequest));
	}

	void swap() {
		recvBuffer = data;
	}

	void distribute(MPI_Comm comm) {
		HR(MPI_Bcast(data.data(), data.size(), MPI_DOUBLE, 0, comm));
	}

	void gatherResults(MPI_Comm comm) {
		assert(type == MatrixType::C);
		HR(MPI_Reduce(data.data(), recvBuffer.data(), data.size(), MPI_DOUBLE, MPI_SUM, 0, comm));
		data = recvBuffer;
	}

	unsigned long long int countGeq(double ge_value) {
		assert(type == MatrixType::C);
		unsigned long long int count = 0;
		for(auto row = 0; row < numRows; row++) {
			for(auto col = 0; col < numCols; col++) {
				if(data[row * numCols + col] >= ge_value) {
					count++;
				}
			}
		}
		return count;
	}

private:
	// to increase locality of memory access, A is stored as row-major, B is stored as column-major
	std::vector<double> data;
	std::vector<double> recvBuffer;
	MatrixType type;
	int rowOffset, colOffset;
	int numRows, numCols;
};

class CanonGroup
{
public:
	CanonGroup() {}
	CanonGroup(int m, int n, int k, int pm, int pn, int pk, 
			   int cA, int cB, int myKRank, int firstAColIncl, int firstBRowIncl, MPI_Comm kComm) 
	: cA(cA), cB(cB) {
		s = std::min(pm, pn);

		// myACanonRank = myKRank % (pm * pn / cA);
		// myBCanonRank = myKRank % (pm * pn / cB);

		// myACanonGroup = myKRank / (pm * pn / cA);
		// myBCanonGroup = myKRank / (pm * pn / cB);

		// Create one communicator for distributing A or B to all cannon groups
		// Create another communicator for sending A and B parts in cannon step
		if (cA > cB)
		{
			myACanonRank = myKRank % (pm * pn / cA);
			myACanonGroup = myKRank / (pm * pn / cA);
			myBCanonRank = myACanonRank;
			myBCanonGroup = myACanonGroup;
			HR(MPI_Comm_split(kComm, myACanonRank, myACanonGroup, &distributeComm));
			HR(MPI_Comm_split(kComm, myACanonGroup, myACanonRank, &canComm));
		}
		else
		{
			myBCanonRank = myKRank % (pm * pn / cB);
			myBCanonGroup = myKRank / (pm * pn / cB);
			myACanonRank = myBCanonRank;
			myACanonGroup = myBCanonGroup;
			HR(MPI_Comm_split(kComm, myBCanonRank, myBCanonGroup, &distributeComm));
			HR(MPI_Comm_split(kComm, myBCanonGroup, myBCanonRank, &canComm));
		}
		
		A = Fragment(
			MatrixType::A,
			(m / s) * (myACanonRank % s), // how many rows per process * my position in a row
			firstAColIncl + (k / (pk * s)) * (myACanonRank / s), // k group offset + how many columns per process in k group * my position in a cannon group square
			(m / s),
			(k / (pk * s)));

		B = Fragment(
			MatrixType::B,
			firstBRowIncl + (k / (pk * s)) * (myBCanonRank % s), // k group offset + how many rows per process in k group * my position in a cannon group square
			(n / s) * (myBCanonRank / s), // how many columns per process * my position in a column
			(k / (pk * s)),
			(n / s));

		C = Fragment(
			MatrixType::C,
			m / s * (myACanonRank % s),
			n / s * myBCanonRank / s,
			(m / (s * cB)),
			(n / (s * cA))
		);
	}

	void run(int seedA, int seedB) {
		generate(seedA, seedB);
		redistribute();
		for(int i = 0; i < s; i++) {
			mulStep();
			sendStep();
		}
	}

	void gatherResults(MPI_Comm comm) {
		C.gatherResults(comm);
	}

	unsigned long long int countGeq(double ge_value) {
		return C.countGeq(ge_value);
	}

private:
	int cA, cB;
	int s;
	int myACanonRank, myBCanonRank;
	int myACanonGroup, myBCanonGroup;
 	Fragment A, B, C;
	MPI_Comm canComm;
	MPI_Comm distributeComm;
	MPI_Request sendRequest[2];
	MPI_Request recvRequest[2];

	void generate(int seedA, int seedB) {
		// Only generate if we are in the first cannon group
		if(myACanonGroup == 0) {
			A.generate(seedA);
		}
		if(myBCanonGroup == 0) {
			B.generate(seedB);
		}
	}

	void redistribute() 
	{
		// this is not asynchronus since we must wait untill all processes have received their data
		if(cA > 0){
			A.distribute(distributeComm);
		} else if (cB > 0){
			B.distribute(distributeComm);
		}
	}

	void mulStep() {
		C.mul(A, B);
	}

	void sendStep() {
		if(s > 1) {
			int sendATo = (myACanonRank + s*(s - 1)) % (s*s);
			int sendBTo = (myBCanonRank + s*s - 1) % (s*s);

			A.sendTo(sendATo, canComm, &sendRequest[0]);
			B.sendTo(sendBTo, canComm, &sendRequest[1]);

			int recvAFrom = (myACanonRank + s) % (s*s);
			int recvBFrom = (myBCanonRank + 1) % (s*s);

			A.recvFrom(recvAFrom, canComm, &recvRequest[0]);
			B.recvFrom(recvBFrom, canComm, &recvRequest[1]);

			HR(MPI_Waitall(2, sendRequest, MPI_STATUSES_IGNORE));
			HR(MPI_Waitall(2, recvRequest, MPI_STATUSES_IGNORE));

			A.swap();
			B.swap();
		}
	}
};

class KTask
{
public:
	KTask(int m, int n, int k, int pm, int pn, int pk, int myRank, MPI_Comm newworld)
	: pk(pk) {
		myKRank = myRank % (pm * pn);
		myKGroup = myRank / (pm * pn);
		MPI_Comm_split(newworld, myKRank, myKGroup, &outsideKComm);
		
		int s = std::min(pm, pn);
		int myKCol = myKRank / s;
		int myKRow = myKRank % s;
		MPI_Comm_split(newworld, myKGroup, myKRank, &insideKComm);
		// communicator for printing results, only used by k group 0
		MPI_Comm_split(newworld, myKGroup, myKCol * s + myKRow, &resultPrintComm);

		int cA = pn > pm ? pn / pm : 1;
		int cB = pm > pn ? pm / pn : 1;
		// int cA = pn > pm ? pn / std::min(pm, pk) : 1;
		// int cB = pm > pn ? pm / std::min(pn, pk) : 1;

		int firstAColIncl = myKGroup * k / pk;
		int firstBRowIncl = myKRank * k / pk;

		canonGroup = CanonGroup(m, n, k, pm, pn, pk, cA, cB, myKRank, firstAColIncl, firstBRowIncl, insideKComm);
	}

	void run(int seedA, int seedB) {
		canonGroup.run(seedA, seedB);
	}

	void gatherResults() {
		if(pk > 1) {
			std::cout << "k group: " << myKGroup << " k rank: " << myKRank << std::endl;
			std::cout << "pk > 1" << std::endl;
			canonGroup.gatherResults(outsideKComm);
		}
	}

	void countGeq(double ge_value) {
		if(myKGroup == 0) {
			unsigned long long int count = canonGroup.countGeq(ge_value);
			unsigned long long int result;
			HR(MPI_Reduce(&count, &result, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, resultPrintComm));
			if(myKRank == 0) {
				std::cout << result << std::endl;
			}
		}
	}

private:
	int myKRank;
	int myKGroup;
	int pk;
	MPI_Comm outsideKComm;
	MPI_Comm insideKComm;
	MPI_Comm resultPrintComm;
	CanonGroup canonGroup;
};

class MatrixProcess
{
public:
	MatrixProcess(PArgs pargs, int pm, int pn, int pk, int myRank, MPI_Comm newworld)
	: pargs(pargs), pm(pm), pn(pn), pk(pk), myRank(myRank),
	  ktask(pargs.m, pargs.n, pargs.k, pm, pn, pk, myRank, newworld), newworld(newworld) {
	}

	void run() {
		for(auto seed : pargs.seeds) {
			ktask.run(seed.first, seed.second);
			if(pargs.g_flag) {
				ktask.gatherResults();
				ktask.countGeq(pargs.ge_value);
			}
		}
	}

private:
	PArgs pargs;
	int pm, pn, pk;
	int myRank;
	KTask ktask;
	MPI_Comm newworld;
};



#endif /* __CA3DMM_H__ */