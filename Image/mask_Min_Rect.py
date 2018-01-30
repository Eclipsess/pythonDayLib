import numpy as np
ms = np.array([[[1,2,3],
              [0,2,0],
              [0,2,3]],
              [[0,1,2],
              [0,3,0],
              [1,2,0]]])
              
print m.shape
m = ms[:,:,0]
print m

print np.any(m, axis=0)
print np.any(m, axis=1)

print np.where(np.any(m, axis=0))[0][[0,-1]]
print np.where(np.any(m, axis=1))

x0, x1 = np.where(np.any(m, axis=0))[0][[0,-1]]
y0, y1 = np.where(np.any(m, axis=1))[0][[0,-1]]
