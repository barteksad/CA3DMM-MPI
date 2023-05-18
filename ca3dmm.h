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
			return std::make_tuple(pm, pn, pk);
		}
	}

	assert(false);
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

		// std::cout<< "A: " << A.numRows << " " << A.numCols << " B: " << B.numRows << " " << B.numCols << "\n";
		assert(A.numCols == B.numRows);

		for(auto row = 0; row < numRows; row++) {
			for(auto col = 0; col < numCols; col++) {
				double sum = 0;
				for(auto idx = 0; idx < A.numCols; idx++) {
					sum += A.data[row * A.numCols + idx] * B.data[col * B.numRows + idx];
				}
				data[row * numCols + col] += sum;
			}
		}
		
	}

	void sendTo(int dest, MPI_Comm comm, MPI_Request *sendRequest) {
		HE(MPI_Isend(data.data(), data.size(), MPI_DOUBLE, dest, 0, comm, sendRequest));
	}

	void recvFrom(int src, MPI_Comm comm, MPI_Request *recvRequest) {
		HE(MPI_Irecv(recvBuffer.data(), recvBuffer.size(), MPI_DOUBLE, src, 0, comm, recvRequest));
	}

	void swap() {
		recvBuffer = data;
	}

	void distribute(MPI_Comm comm) {
		HE(MPI_Bcast(data.data(), data.size(), MPI_DOUBLE, 0, comm));
	}

	void gatherResults(MPI_Comm comm) {
		assert(type == MatrixType::C);
		HE(MPI_Reduce(data.data(), recvBuffer.data(), data.size(), MPI_DOUBLE, MPI_SUM, 0, comm));
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

	void printRow(int rowNum, bool endline) {
		assert(type == MatrixType::C);
		for(auto col = 0; col < numCols; col++) {
			std::cout << data[rowNum * numCols + col];
			if(col == numCols - 1 && endline) {
				std::cout << "\n";
			} else {
				std::cout << " ";
			}
		}
	}

	void sendRow(int rowNum, MPI_Comm comm) {
		assert(type == MatrixType::C);
		HE(MPI_Send(data.data() + rowNum * numCols, numCols, MPI_DOUBLE, 0, 0, comm));
	}

	void recvPrintRow(int src, MPI_Comm comm, bool endline) {
		assert(type == MatrixType::C);
		HE(MPI_Recv(recvBuffer.data(), numCols, MPI_DOUBLE, src, 0, comm, MPI_STATUS_IGNORE));
		for(auto col = 0; col < numCols; col++) {
			std::cout << recvBuffer[col];
			if(col == numCols - 1 && endline) {
				std::cout << "\n";
			} else {
				std::cout << " ";
			}
		}
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

		// Create one communicator for distributing A or B to all cannon groups
		// Create another communicator for sending A and B parts in cannon step
		myCanonRank = myKRank % (pm * pn / std::max(cA, cB));
		myCanonGroup = myKRank / (pm * pn / std::max(cA, cB));
		HE(MPI_Comm_split(kComm, myCanonRank, myCanonGroup, &distributeComm));
		HE(MPI_Comm_split(kComm, myCanonGroup, myCanonRank, &canComm));

		int rankCanon, rankDistribute;
		MPI_Comm_rank(canComm, &rankCanon);
		MPI_Comm_rank(distributeComm, &rankDistribute);
		
		A = Fragment(
			MatrixType::A,
			(m / pm), // how many rows per process * my position in a row
			firstAColIncl + ((k / pk) / (pn / cA)) * (myCanonRank / pm), // offset + per k task / how many columns per process  * my column position
			(m / pm),
			(k / pk) / (pn / cA));

		B = Fragment(
			MatrixType::B,
			firstBRowIncl + ((k / pk) / (pm / cB)) * (myCanonRank % pm), // offset + per k task / how many rows per process * my row position
			(n / pn), // how many columns per process * my position in a column
			(k / pk) / (pm / cB),
			(n / pn));

		C = Fragment(
			MatrixType::C,
			0, // offset does not matter for C
			0, // offset does not matter for C
			(m / pm),
			(n / pn)
		);
	}

	void run(int seedA, int seedB) {
		redistribute();
		for(int i = 0; i < s; i++) {
			mulStep();
			sendStep();
		}
	}

	void generate(int seedA, int seedB) {
		// Only generate if we are in the first cannon group
		if(myCanonGroup == 0 || cA == 1) {
			A.generate(seedA);
		}
		if(myCanonGroup == 0 || cB == 1) {
			B.generate(seedB);
		}
	}

	void gatherResults(MPI_Comm comm) {
		C.gatherResults(comm);
	}

	unsigned long long int countGeq(double ge_value) {
		return C.countGeq(ge_value);
	}

	void printRow(int rowNum, bool endline) {
		C.printRow(rowNum, endline);
	}

	void sendRow(int rowNum, MPI_Comm comm) {
		C.sendRow(rowNum, comm);
	}

	void recvPrintRow(int src, MPI_Comm comm, bool endline) {
		C.recvPrintRow(src, comm, endline);
	}

private:
	int cA, cB;
	int s;
	int myCanonRank;
	int myCanonGroup;
 	Fragment A, B, C;
	MPI_Comm canComm;
	MPI_Comm distributeComm;
	MPI_Request sendRequest[2];
	MPI_Request recvRequest[2];

	void redistribute() 
	{
		// this is not asynchronus since we must wait untill all processes have received their data
		if(cA > 1){
			A.distribute(distributeComm);
		} else if (cB > 1){
			B.distribute(distributeComm);
		}
	}

	void mulStep() {
		C.mul(A, B);
	}

	void sendStep() {
		if(s > 1) {
			int sendATo = (myCanonRank + s*(s - 1)) % (s*s);
			int sendBTo = myCanonRank % s == 0 ? myCanonRank + s - 1 : myCanonRank - 1;

			A.sendTo(sendATo, canComm, &sendRequest[0]);
			B.sendTo(sendBTo, canComm, &sendRequest[1]);

			int recvAFrom = (myCanonRank + s) % (s*s);
			int recvBFrom = (myCanonRank + 1) % s == 0 ? myCanonRank - s + 1 : myCanonRank + 1;

			A.recvFrom(recvAFrom, canComm, &recvRequest[0]);
			B.recvFrom(recvBFrom, canComm, &recvRequest[1]);

			HE(MPI_Waitall(2, sendRequest, MPI_STATUSES_IGNORE));
			HE(MPI_Waitall(2, recvRequest, MPI_STATUSES_IGNORE));

			A.swap();
			B.swap();
		}
	}
};

class KTask
{
public:
	KTask(int m, int n, int k, int pm, int pn, int pk, int myRank, MPI_Comm newworld)
	: n(n), m(m), k(k), pm(pm), pn(pn), pk(pk) {
		myKRank = myRank % (pm * pn);
		myKGroup = myRank / (pm * pn);
		// for gathering results
		MPI_Comm_split(newworld, myKRank, myKGroup, &outsideKComm);
		
		// int s = std::min(pm, pn);
		int myKCol = myKRank / pm;
		int myKRow = myKRank % pm;
		MPI_Comm_split(newworld, myKGroup, myKRank, &insideKComm);
		// communicator for printing results, only used by k group 0
		myPrintRank =  myKRow * pn + myKCol;
		MPI_Comm_split(newworld, myKGroup, myPrintRank, &resultPrintComm);

		int rankOutside, rankInside, rankPrint;
		MPI_Comm_rank(outsideKComm, &rankOutside);
		MPI_Comm_rank(insideKComm, &rankInside);
		MPI_Comm_rank(resultPrintComm, &rankPrint);


		int cA = pn > pm ? pn / pm : 1;
		int cB = pm > pn ? pm / pn : 1;

		int firstAColIncl = myKGroup * k / pk;
		int firstBRowIncl = myKRank * k / pk;

		canonGroup = CanonGroup(m, n, k, pm, pn, pk, cA, cB, myKRank, firstAColIncl, firstBRowIncl, insideKComm);
	}

	void run(int seedA, int seedB) {
		if(myKGroup == 0) {
			canonGroup.generate(seedA, seedB);
		}
		canonGroup.run(seedA, seedB);
	}

	void gatherResults() {
		if(pk > 1) {
			canonGroup.gatherResults(outsideKComm);
		}
	}

	void countGeq(double ge_value) {
		if(myKGroup == 0) {
			unsigned long long int count = canonGroup.countGeq(ge_value);
			unsigned long long int result;
			HE(MPI_Reduce(&count, &result, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, resultPrintComm));
			if(myKRank == 0) {
				std::cout << result << "\n";
			}
		}
	}

	void printResults() {
		if(myKGroup == 0) {
			for(int rowProc = 0; rowProc < pm; rowProc++) {
				for(int rowNum = 0; rowNum < m / pm; rowNum++) {
					for(int colProc = 0; colProc < pn; colProc++) {
						int curr = rowProc * pn + colProc;
						bool endline = colProc == pn - 1;
						if(curr == 0 && myPrintRank == 0) {
							canonGroup.printRow(rowNum, endline);
						} else if(myPrintRank == (rowProc * pn + colProc)) {
							canonGroup.sendRow(rowNum, resultPrintComm);
						} else if(myPrintRank == 0) {
							canonGroup.recvPrintRow(rowProc * pn + colProc, resultPrintComm, endline);
						}
						HE(MPI_Barrier(resultPrintComm));
					}
				}
			}
		}
	}

private:
	int myKRank;
	int myKGroup;
	int myPrintRank;
	int n, m, k;
	int pm ,pn, pk;
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
			} else if(pargs.v_flag) {
				if(myRank == 0) {
					std::cout << pargs.m << " " << pargs.n << "\n";
				}
				ktask.gatherResults();
				ktask.printResults();
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