import numpy as np 

m = np.array([

    [5, 2, 3],
    [1, 1, 11],
    [3, 4, 55]

])


## flip the matrix horizontally. ie.. left to right

m_horizontally = m[: ,::-1]

print(m_horizontally)