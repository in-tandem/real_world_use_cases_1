import numpy as np 
from compare_two_matrices import compare
m = np.array([

    [5, 2, 3],
    [1, 1, 11],
    [3, 4, 55]

])


## flip the matrix horizontally. ie.. left to right

m_horizontally = m[: ,::-1]

print(m_horizontally)

assert compare(m_horizontally, np.fliplr(m))

random_m = np.random.rand(4,5)

assert compare(random_m[:,::-1], np.fliplr(random_m))

## flip matrix vertically

m_vertically = m[::-1]
print(m_vertically)
assert compare(random_m[::-1], np.flipud(random_m))
assert compare(m_vertically, np.flipud(m))


## lets combine both to get the matrix flipped horizontally and vertically as well

assert compare( random_m[:,::-1][::-1], np.flipud(np.fliplr(random_m)))