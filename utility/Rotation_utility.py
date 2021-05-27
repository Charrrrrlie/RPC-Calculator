import numpy as np
import math
from scipy.spatial.transform import Rotation as R

class Rotation:

    def __init__(self,path_Ru,path_Direct,path_rot,exter_element):

        #相机到本体坐标系 校准矩阵
        self.Ru=self.load_Ru(path_Ru)

        #本体坐标系到J2000坐标系 shape(N_time,3,3)
        self.R_b2j=R.from_quat(exter_element.att).as_matrix()

        #J2000坐标系到WGS84坐标系 shape(N_time,3,3)
        self.R_j2w=self.load_Rj2w(path_rot)

        #指向角文件 shape(N_time,2)
        direct_data=self.load_direct_data(path_Direct)

        if self.Ru==[] or self.R_b2j==[] or self.R_j2w==[] or len(direct_data)==0:
            print("读入旋转矩阵相关数据出错")
        else:
            #初始视向量 shape(3,N_pix)
            self.ux=self.get_ux(direct_data)

    def load_direct_data(self,dir):
        direct = np.loadtxt(dir, usecols=(1, 2), skiprows=1)
        return direct

    def load_Ru(self,dir):
        f = open(dir)
        data = f.readlines()
        f.close()

        #pitch  phi
        p = float(data[1].split('=', 1)[1])
        #roll  omega
        r = float(data[3].split('=', 1)[1])
        #yaw   kappa
        y = float(data[5].split('=', 1)[1])

        R_phi=np.array([
                         [math.cos(p),0,-math.sin(p)],
                         [  0 ,       1,      0     ],
                         [math.sin(p), 0, math.cos(p)],
                        ])
        R_omega=np.array([
                         [  1 ,       0 ,      0      ],
                         [  0 ,math.cos(r), -math.sin(r)],
                         [  0 ,math.sin(r), math.cos(r)],
                        ])
        R_kappa=np.array([
                         [math.cos(y), -math.sin(y) ,0],
                         [math.sin(y),  math.cos(y) ,0],
                         [0, 0, 1],
                        ])

        return np.dot(np.dot(R_phi,R_omega),R_kappa)

    def load_Rj2w(self,dir):
        f = open(dir)
        data=f.readlines()
        f.close()

        res=[]
        for i in data:
            res.append(i.split(" ")[0:-1])
        res=np.array(res).reshape(-1,3,3).astype(np.float)
        return res

    def get_ux(self,directData):
        """
        :param directData: 指向角文件 id 垂轨指向角 沿轨指向角
        :return: 旋转前的视向量ux  3*N
        """
        length=len(directData)
        return np.concatenate([np.tan(directData[:,1]).reshape(1,length)
                              ,np.tan(directData[:,0]).reshape(1,length)
                               ,-np.ones((1,length))])

    #解二次函数
    def result(self,a, b, c):
        derat = b ** 2 - 4 * a * c
        if a == 0:
            if b != 0:
                x = -c / b
                return x
            else:
                return '无解 '
        else:
            if derat < 0:
                return '无实根 '
            elif derat == 0:
                x = -b / (2 * a)
                return x
            else:
                xone = (-b + math.sqrt(derat)) / (2 * a)
                xtwo = (-b - math.sqrt(derat)) / (2 * a)
                return max(xone, xtwo)

    #比例系数
    def get_M(self,UX,PX,h):
        """
        :param UX,PX: 旋转后的视向量UX,卫星坐标
        :return: 比例系数与焦距的乘积m=u*f
        """
        # WGS84椭球参数
        A=6378137+h
        B=6356752.3142+h
        #二次函数系数分别为a,b,c
        a=(UX[0]*UX[0]+UX[1]*UX[1])/(A*A)+UX[2]*UX[2]/(B*B)
        b=2*((UX[0]*PX[0]+UX[1]*PX[1])/(A*A)+UX[2]*PX[2]/(B*B))
        c=(PX[0]*PX[0]+PX[1]*PX[1])/(A*A)+PX[2]*PX[2]/(B*B)-1
        u=self.result(a,b,c)
        if len(u)==1:
            #print(u)
            m=u
        else:
            print('无解 a:{} b:{} c:{}'.format(a,b,c))
            m=0 #随便赋了个值
        return m

    def get_XYZ(self,ux,R1,R2,PX,h):
        """
        :param ux:旋转前的视向量ux
        :param R1:本体坐标系到J2000坐标系的转换
        :param R2:J2000坐标系到WGS84坐标系的转换
        :param Ru:相机坐标系到本体坐标系
        :param PX:卫星GPS位置
        :param h:虚拟格网点高程
        :return:像点对应地面坐标XYZ
        """

        # R2=np.array([[-0.6215, -0.7834 , 0.0008],
        #             [0.7834, -0.6215, -0.0010],
        #             [0.0013, -0.0000, 1.0000]])

        #f=0.043
        #ux*=f

        ux=ux.reshape(-1,1)
        PX=PX.reshape(-1,1)

        U=np.dot(self.Ru,ux)#中间变量U
        U=np.dot(R1,U)
        UX=np.dot(R2,U) #旋转后的视向量

        m=self.get_M(UX,PX,h)
        U=UX*m
        XYZ=PX+U

        return XYZ

    def XYZ2BLH(self,XYZ):
        """
        :param XYZ: WGS84坐标系坐标
        :return: BLH：经纬度大地高
        """
        a = 6378137
        b = 6356752.314
        pi= 3.1415926
        e2 = (a * a - b * b) / (a * a)
        L = np.arctan(XYZ[:, 1] / XYZ[:, 0])
        B = np.arctan(XYZ[:, 2] / np.sqrt(XYZ[:, 0] ** 2 + XYZ[:, 1] ** 2))

        for i in range(0, 100):
            N = a / np.sqrt(1 - e2 * (np.sin(B) ** 2))
            H = XYZ[:, 2] / np.sin(B) - N * (1 - e2)

            Bn = np.arctan(XYZ[:, 2] * (N + H) / ((N * (1 - e2) + H) * np.sqrt(XYZ[:, 0] ** 2 + XYZ[:, 1] ** 2)))

            if np.max(np.abs(B - Bn)) < 1e-7:
                print('successful transform！')
                break
            B = Bn

        BLH = np.zeros((len(B), 3))

        #记得转换为角度！！！！！！！！
        B=B / pi * 180
        L=L / pi * 180

        B[B<0]+=180
        L[L<0]+=180

        BLH[:, 0] = B
        BLH[:, 1] = L
        BLH[:, 2] = H

        return BLH

    def XYZ2BLH2(self,XYZ):
        """
        第二种转换方式
        :param XYZ: WGS84坐标系坐标
        :return: BLH：经纬度大地高
        """

        a = 6378137
        b = 6356752.314
        pi = 3.1415926

        e2 = (a * a - b * b) / (a * a)
        L = np.arctan(XYZ[:, 1] / XYZ[:, 0])
        B = np.arctan(XYZ[:, 2] / np.sqrt(XYZ[:, 0] ** 2 + XYZ[:, 1] ** 2))

        for i in range(0, 100):
            N = a / np.sqrt(1 - e2 * (np.sin(B) ** 2))
            H = np.sqrt(XYZ[:,0]**2 + XYZ[:, 1]**2) / np.cos(B) - N
            Bn = np.arctan((XYZ[:, 2]+ N * e2 * np.sin(B)) / (np.sqrt(XYZ[:, 0] ** 2 + XYZ[:, 1] ** 2)))

            if np.max(np.abs(B - Bn)) < 1e-7:
                print('successful transform！')
                break
            B = Bn

        BLH = np.zeros((len(B), 3))

        # 记得转换为角度！！！！！！！！
        B = B / pi * 180
        L = L / pi * 180

        B[B < 0] += 180
        L[L < 0] += 180

        BLH[:, 0] = B
        BLH[:, 1] = L
        BLH[:, 2] = H
        return BLH

    def BLH2XYZ(self,BLH):
        a = 6378137.0000
        b = 6356752.3141
        pi = 3.1415926
        e2 = 1 - (b / a) ** 2

        B=BLH[:,0]
        L=BLH[:,1]
        H=BLH[:,2]

        L = L * pi / 180
        B = B * pi / 180

        N = a / np.sqrt(1 - e2 * np.sin(B) ** 2)
        x = (N + H) * np.cos(B) * np.cos(L)
        y = (N + H) * np.cos(B) * np.sin(L)
        z = (N * (1 - e2) + H) * np.sin(B)
        return np.vstack((x,y,z)).transpose()