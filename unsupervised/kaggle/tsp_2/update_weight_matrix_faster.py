import numpy as np 
from multiprocessing import Pool

a = np.random.randint(1,100, (100,2))
a = a.tolist()
# a = [[22,2],[33,3],[44,4],[5,55]]
a.append([11,22])
a.append([22,31])

print(type(a),a)
b = [[11,22],[22,31],[22,22],[11,1]]

def not_in(array, another_array):
        return [i for i in array if i not in another_array]
        
if __name__ =='__main__':
    final = [[0,0]]

    with Pool(8) as p:

        added = p.starmap(not_in, [(i.tolist(),b) for i in np.array_split(a,8)])
        print(added)
        final = np.concatenate((final,added[0] if len(added)>0 else [0,0]), axis=0)
        # final.extend(p.starmap(not_in, [(i.tolist(),b) for i in np.array_split(a,2)]))

    print(final)