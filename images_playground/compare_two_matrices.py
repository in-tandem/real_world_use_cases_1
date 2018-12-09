import numpy as np 


def compare(m1,m2):

    flag = False

    if m1.shape == m2.shape:

        for i,j in enumerate(m1):
            
            check_all = list(set(filter(lambda x :True if x else False,[i==k for i,k in zip(j,m2[i])])))

            if check_all and len(check_all)==1 and check_all[0]:

                continue
            
            else:
                return flag

        flag = True
    
    return flag

print(compare(np.random.rand(2,3),np.random.rand(2,3)))

print(compare(np.array([[1,2],[2,3]]),np.array([[1,2],[2,3]])))
m = np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]])
w = np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]])
print(compare(m, w))

print((m==w).all())