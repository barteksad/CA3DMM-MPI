#include <iostream>
#include <vector>

#include "../densematgen.h"
#include "../utils.h"

using namespace std;

using mat_t = vector<vector<double>>;

int main(int argc, char **argv)
{
  PArgs args = parse_args(argc, argv);

  for (auto seed : args.seeds)
  {
    int n = args.n, m = args.m, k = args.k;
    
    mat_t A(m, vector<double>(k, 0));
    mat_t B(k, vector<double>(n, 0));
    mat_t C(m, vector<double>(n, 0));

    for (int i = 0; i < m; i++)
      for (int j = 0; j < k; j++)
        A[i][j] = generate_double(seed.first, i, j);

    for (int i = 0; i < k; i++)
      for (int j = 0; j < n; j++)
        B[i][j] = generate_double(seed.second, i, j);

    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        for (int l = 0; l < k; l++)
          C[i][j] += A[i][l] * B[l][j];

    if(args.g_flag) 
    {
      unsigned long long int count = 0;
      for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
          if(C[i][j] >= args.ge_value)
            count++; 

      cout << count << endl;
    }
    else
    {
      std::cout << m << " " << n << "\n";
      for (int i = 0; i < m; i++)
      {
        for (int j = 0; j < n; j++)
          std::cout << C[i][j] << " ";
        std::cout << std::endl;
      }
    }
  }
}