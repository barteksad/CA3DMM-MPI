import sys
import numpy as np

try:
	file1 = open(sys.argv[1], 'r')
	file2 = open(sys.argv[2], 'r')
except Exception as e:
	print("Usage: python check_output.py <file1> <file2>")
	exit(1)
        
lines1 = file1.readlines()[1:-1]
lines2 = file2.readlines()[1:-1]

C1 = np.zeros((len(lines1), len(lines1[0].split())))
C2 = np.zeros((len(lines2), len(lines2[0].split())))

for i in range(len(lines1)):
    C1[i] = np.array(lines1[i].split(), dtype=np.double)

for i in range(len(lines2)):
    C2[i] = np.array(lines2[i].split(), dtype=np.double)
    
print(np.allclose(C1, C2, atol=1e-5))