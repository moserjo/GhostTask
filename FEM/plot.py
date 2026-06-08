import os
import numpy as np
import matplotlib.pyplot as plt


data_path = os.path.join(
    os.path.dirname(__file__),
    "Ex3_test2.npz"
)
data = np.load(data_path)
for key in data:
    print(key)
A = data["A"]
B = data["B"]
#J = data["J"]
centroids = data["centroids"]
mag = 20
print(A.shape, B.shape, centroids.shape) #wayyy too many points, need a way to specify on which points to evaluate: daweil 6 centroids in einer ebene um den jeweiligen grid punkt herum
x = centroids[:,0]
y = centroids[:,1]
z = centroids[:,2]
Ax = 0
Ay = 0
Az = np.sin(np.pi*x)*np.sin(np.pi*y)#mag*x*(1-x)*y*(1-y)*(z-1)*(2-z)

print(np.max(abs(Ax - A[:,0])),np.max(A[:,0]))
print(np.max(abs(Ay - A[:,1])), np.max(A[:,1]))
print(np.max(abs(Az - A[:,2])), np.max(A[:,2]))
Bx = np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)#mag*x*(1-x)*(1-2*y)*(z-1)*(2-z)
By = -np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)#-mag*(1-2*x)*y*(1-y)*(z-1)*(2-z)
Bz = 0
print(np.max(abs(Bx -  B[:,0])), np.max(B[:,0]))
print(np.max(abs(By - B[:,1])), np.max(B[:,1]))
print(np.max(abs(Bz - B[:,2])), np.max(B[:,2]))

Jx = 0#mag*(1 - 2*x) * y*(1-y) * (3 - 2*z)
Jy = 0#mag*x*(1-x) * (1 - 2*y) * (3 - 2*z)
Jz = 0#mag*2 * (z-1)*(2-z) * (x*(1-x) + y*(1-y))
#print(J)
"""print(np.max(abs(Jx -  J[:,0])), np.max(J[:,0]))
print(np.max(abs(Jy - J[:,1])), np.max(J[:,1]))
print(np.max(abs(Jz - J[:,2])), np.max(J[:,2]))"""
k = 5
xx = x[12*12*(k-1):12*12*k]
yy = y[12*12*(k-1):12*12*k]

Bxx = B[12*12*(k-1):12*12*k, 2]
plt.tricontourf(xx, yy, Bxx, levels=30)
plt.colorbar()
plt.show()

fig = plt.figure()#projection = "3d")
ax = fig.add_subplot(projection='3d')
#for i in range(25*5):
ax.scatter(centroids[:12*12,0], centroids[:12*12,1], centroids[:12*12,2])#
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z") #conclusio: jeweils die nxy*nxy*nz ersten points ergeben ein cubic mesh, will use that so save: centroids[:sol.n_xy*sol.n_xy*sol.n_z]
plt.legend()
plt.show()