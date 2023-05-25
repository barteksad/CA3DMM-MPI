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
	int best = std::numeric_limits<int>::max();
	int bestPm=-1, bestPn=-1, bestPk=-1;

	for(int pm = 1; pm <= p; pm ++)
	{
		for(int pn = 1; pn <= p; pn ++)
		{
			for(int pk = 1; pk <= p; pk ++)
			{
				int nProc = pm * pn * pk;
				if(nProc > p || nProc < p * 0.95) continue;
				if(std::max(pm, pn) % std::min(pm, pn) != 0) continue;

				int comCost = pk * m * n + pm * n * k + pn * k * m;
				if(comCost <= best)
				{
					best = comCost;
					bestPm = pm;
					bestPn = pn;
					bestPk = pk;
				}
			}
		}
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
					// if(data[row * numCols + col] > 0) {
					// 	std::cout << "A[" << row + rowOffset << ", " << col + colOffset << "] = " << data[row * numCols + col] << std::endl;
					// }
				}
			}
		}
		else
		{
			for(auto col = 0; col < numCols; col++) {
				for(auto row = 0; row < numRows; row++) {
					data[col * numRows + row] = generate_double(seed, row + rowOffset, col + colOffset);
					// if(data[col * numRows + row] > 0) {
					// 	std::cout << "B[" << row + rowOffset << ", " << col + colOffset << "] = " << data[col * numRows + row] << std::endl;
					// }
				}
			}
		}
	}

	void mul(const Fragment& A, const Fragment& B) {
		assert(type == MatrixType::C);
		assert(A.type == MatrixType::A);
		assert(B.type == MatrixType::B);

		assert(A.numCols == B.numRows);

		for(auto row = 0; row < numRows; row++) {
			for(auto col = 0; col < numCols; col++) {
				double sum = 0;
				for(auto idx = 0; idx < A.numCols; idx++) {
					sum += A.data[row * A.numCols + idx] * B.data[col * B.numRows + idx];
				}
				// if(sum > 0)
				// 	std::cout << "C[" << row + A.rowOffset << "," << col + B.colOffset << "] += " << sum << "\n";
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
		data = recvBuffer;
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

	void checkSendRow(int rowNum, int colNum, MPI_Comm comm) {
		assert(type == MatrixType::C);
		if(rowNum >= rowOffset && rowNum < rowOffset + numRows && colNum >= colOffset && colNum < colOffset + numCols) {
			rowNum %= numRows;
			colNum %= numCols;
			HE(MPI_Send(data.data() + rowNum * numCols, numCols, MPI_DOUBLE, 0, 0, comm));
		}
	}

	void recvPrintRow(MPI_Comm comm, bool endline) {
		assert(type == MatrixType::C);
		HE(MPI_Recv(recvBuffer.data(), numCols, MPI_DOUBLE, MPI_ANY_SOURCE, 0, comm, MPI_STATUS_IGNORE));
		for(auto col = 0; col < numCols; col++) {
			std::cout << recvBuffer[col];
			if(col == numCols - 1 && endline) {
				std::cout << "\n";
			} else {
				std::cout << " ";
			}
		}
	}

// private:
public:
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
			// cB == 1 ? (m / (pk * pm)) * (myCanonRank % pm) : (m / (pk * pn)) * myCanonGroup + (m / (pk * pn * pn)) * (myCanonRank % pn), // how many rows per process * my position in a row
			cB == 1 ? (m / pm) * (myCanonRank % pm) : (m / cB) * myCanonGroup + (m / (cB * pn)) * (myCanonRank % pn), // how many rows per process * my position in a row
			firstAColIncl + (k / (pk * s)) * (myCanonRank / s), // offset + per k task / how many columns per process  * my column position
			(m / pm),
			(k / (pk * s)));

		B = Fragment(
			MatrixType::B,
			firstBRowIncl + (k / (pk * s)) * (myCanonRank % s), // offset + per k task / how many rows per process * my row position
			// (n / (pn / cA)) * myCanonGroup + (n / pn) * (myCanonRank / s), // how many columns per process * my position in a column
			cA == 1 ? (n / pn) * (myCanonRank / pn) : (n / cA) * myCanonGroup + (n / (cA * pm)) * (myCanonRank / pm), // how many columns per process * my position in a column
			(k / (pk * s)),
			(n / pn));

		C = Fragment(
			MatrixType::C,
			cB == 1 ? (m / pm) * (myCanonRank % pm) : (m / cB) * myCanonGroup + (m / (cB * pn)) * (myCanonRank % pn),
			cA == 1 ? (n / pn) * (myCanonRank / pn) : (n / cA) * myCanonGroup + (n / (cA * pm)) * (myCanonRank / pm),
			(m / pm),
			(n / pn)
		);

				// std::cout << "My kRank: " << myKRank << "\n"
				//   << "My canon rank: " << myCanonRank << "\n"
				//   << "My canon group: " << myCanonGroup << "\n"
				//   << "A row offset: " << A.rowOffset << "\n"
				//   << "A col offset: " << A.colOffset << "\n"
				//   << "A num rows: " << A.numRows << "\n"
				//   << "A num cols: " << A.numCols << "\n"
				//   << "B row offset: " << B.rowOffset << "\n"
				//   << "B col offset: " << B.colOffset << "\n"
				//   << "B num rows: " << B.numRows << "\n"
				//   << "B num cols: " << B.numCols << "\n"
				// 	<< "C row offset: " << C.rowOffset << "\n"
				// 	<< "C col offset: " << C.colOffset << "\n"			
				// 	<< "C num rows: " << C.numRows << "\n"
				// 	<< "C num cols: " << C.numCols << "\n"
				//   << " --------------- \n\n";
	}

	void run(int seedA, int seedB) {
		redistribute();
		initialCanonSkewSend();
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

	void checkSendRow(int rowNum, int colNum, MPI_Comm comm) {
		C.checkSendRow(rowNum, colNum, comm);
	}

	void recvPrintRow(MPI_Comm comm, bool endline) {
		C.recvPrintRow(comm, endline);
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

			// std::cout << "Rank " << myCanonRank << " sending A to " << sendATo << " and B to " << sendBTo << "\n" << "receiving A from " << recvAFrom << " and B from " << recvBFrom << "\n\n";

			HE(MPI_Waitall(2, sendRequest, MPI_STATUSES_IGNORE));
			HE(MPI_Waitall(2, recvRequest, MPI_STATUSES_IGNORE));

			A.swap();
			B.swap();
		}
	}

	void initialCanonSkewSend() {
		if(s > 1) {
			MPI_Request waitFor[4];
			int nWaitFor = 0;

			int aSkew = myCanonRank % s;
			int bSkew = myCanonRank / s;
			int bOffset = (myCanonRank / s) * s;

			int sendATo = (myCanonRank + s*(s - aSkew)) % (s*s);
			int sendBTo = (myCanonRank - bOffset + (s - bSkew)) % s + bOffset;

			// std::cout << "init: Rank " << myCanonRank << " sending A to " << sendATo << " and B to " << sendBTo << "\n";

			if(sendATo != myCanonRank) {
				A.sendTo(sendATo, canComm, &sendRequest[0]);
				waitFor[nWaitFor++] = sendRequest[0];
			}
			if(sendBTo != myCanonRank) {
				B.sendTo(sendBTo, canComm, &sendRequest[1]);
				waitFor[nWaitFor++] = sendRequest[1];
			}

			int recvAFrom = (myCanonRank + s*(s + aSkew)) % (s*s);
			int recvBFrom = (myCanonRank - bOffset + bSkew) % s + bOffset;

			// std::cout << "Rank " << myCanonRank << " receiving A from " << recvAFrom << " and B from " << recvBFrom << "\n\n";

			if(recvAFrom != myCanonRank) {
				A.recvFrom(recvAFrom, canComm, &recvRequest[0]);
				waitFor[nWaitFor++] = recvRequest[0];
			}
			if(recvBFrom != myCanonRank) {
				B.recvFrom(recvBFrom, canComm, &recvRequest[1]);
				waitFor[nWaitFor++] = recvRequest[1];
			}

			HE(MPI_Waitall(nWaitFor, waitFor, MPI_STATUSES_IGNORE));

			if(sendATo != myCanonRank) {
				A.swap();
			}
			if(sendBTo != myCanonRank) {
				B.swap();
			}
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
		
		
		MPI_Comm_split(newworld, myKGroup, myKRank, &insideKComm);
		// communicator for printing results, only used by k group 0
		// int myKRow = myKRank % pn;
		// int myKCol = myKRank / pn;
		// myPrintRank =  myKRow * pn + myKCol;
		// MPI_Comm_split(newworld, myKGroup, myPrintRank, &resultPrintComm);
		// std::cout << "Rank " << myRank << " is in k group " << myKGroup << " and has k rank " << myKRank << " and print rank " << myPrintRank << "\n";


		int cA = pn > pm ? pn / pm : 1;
		int cB = pm > pn ? pm / pn : 1;

		int firstAColIncl = myKGroup * k / pk;
		int firstBRowIncl = myKGroup * k / pk;

		canonGroup = CanonGroup(m, n, k, pm, pn, pk, cA, cB, myKRank, firstAColIncl, firstBRowIncl, insideKComm);
	}

	void run(int seedA, int seedB) {
		canonGroup.generate(seedA, seedB);
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
			HE(MPI_Reduce(&count, &result, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, insideKComm));
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
						int colIdx = colProc * n / pn;
						int rowIdx = rowProc * m / pm + rowNum;
						bool endline = colProc == pn - 1;

						if(curr == 0 && myKRank == 0) {
							canonGroup.printRow(rowNum, endline);
						} else if(curr > 0 && myKRank != 0) {
							canonGroup.checkSendRow(rowIdx, colIdx, insideKComm);
						} else if(curr > 0 && myKRank == 0) {
							canonGroup.recvPrintRow(insideKComm, endline);
						}
						HE(MPI_Barrier(insideKComm));
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