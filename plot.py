import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def XYZ2BLH(XYZ):
    """
    :param XYZ: WGS84坐标系坐标
    :return: BLH：经纬度大地高
    """
    a = 6378137
    b = 6356752.314
    pi=3.1415926
    e2 = (a * a - b * b) / (a * a)
    L = np.arctan(XYZ[:, 1] / XYZ[:, 0])
    B = np.arctan(XYZ[:, 2] / np.sqrt(XYZ[:, 0] ** 2 + XYZ[:, 1] ** 2))

    for i in range(0, 100):
        N = a / np.sqrt(1 - e2 * (np.sin(B) ** 2))
        H = XYZ[:, 2] / np.sin(B) - N * (1 - e2)

        Bn = np.arctan(XYZ[:, 2] * (N + H) / ((N * (1 - e2) + H) * np.sqrt(XYZ[:, 0] ** 2 + XYZ[:, 1] ** 2)))

        if np.max(np.abs(B - Bn)) < 1e-7:
            break
        B = Bn

    BLH = np.zeros((len(B), 3))
    BLH[:, 0] = B /pi*180
    BLH[:, 1] = (L /pi*180) +180
    BLH[:, 2] = H

    return BLH

obj=np.load('./intermediate_res/gcp.npy')
gps=np.load('./intermediate_res/gps_loc.npy')

gps=XYZ2BLH(gps)

X=gps[:,0] #121:241
Y=gps[:,1]
Z=gps[:,2]

#print(a)
x=obj[:,0]
y=obj[:,1]
z=obj[:,2]


fig=plt.figure()
ax = Axes3D(fig)
ax.scatter(x[0:121],y[0:121],z[0:121],color='c',linewidths=0.01)
ax.scatter(x[121:242],y[121:242],z[121:242],color='k',linewidths=0.01)
ax.scatter(x[242:363],y[242:363],z[242:363],color='b',linewidths=0.01)

#ax.scatter(x,y,z,color='c',linewidths=0.01)

#卫星位置
#ax.scatter(X,Y,Z,color='r',linewidths=0.01)

plt.show()
#fig.savefig('./figs/gcp.png',bbox_inches='tight')
