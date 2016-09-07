Jeremy Colebrook-Soucie
Comp 193 - GPU Programming
Assignment 1, vector addition

Assignment went well. I successfully added code to implement becotr 
addition on the gpu.

5) For this question, vectors of size 50,000 were used for comparison.
This is because 100,000 is larger than the hardware limit of 65536. 
Exceeding this limit causes failures on the GPU. 
linux time command and time.h both indicate that the CPU took far less time 
than the GPU. For instance, the time command indicates
that the CPU version used 0.002 seconds while the GPU version 0.006 seconds.
This is likely due to superior CPU optimizations. 
