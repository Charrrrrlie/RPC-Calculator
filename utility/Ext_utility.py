import numpy as np

from matplotlib import pyplot as plt

#返回 self.image_time  att gps  三者长度相同

class Ext:

    def __init__(self,attData_dir,gpsData_dir,imgTime_dir):

        self.image_time=self.load_imagetime(imgTime_dir)
        self.gps_data,self.gps_time=self.load_gpsData(gpsData_dir)
        self.att_data,self.att_time=self.load_attData(attData_dir)
        self.transformed_time = self.transform_time(self.image_time)

        if len(self.image_time)==0 or len(self.gps_data)==0 or len(self.att_data)==0:
            print("读入外方位相关数据出错")
        else:
            self.att=self.att_interpolate(self.image_time,self.att_data,self.att_time)
            self.gps_loc,self.gps_v=self.gps_interpolate(self.image_time,self.gps_data,self.gps_time)

    def load_gpsData(self,gpsData_dir):
        """
        load gpsData from DX_ZY3_NAD_gps.txt
        input: gpsData_dir : file path
        return:
            gpsData(array, shape of (groupNumber,6)): PX PY PZ VX VY VZ;
            gpsData_time(array, shape of(groupNumber,1)): timeCode
        """
        data_info = []  # 存放所有从txt文件中读出的数据
        for line in open(gpsData_dir, "r"):
            line = line[:-1]  # 逐行读，去掉每行最后的换行符
            data_info.append(line)
        # A为临时变量，将数据格式从list转为string，为了在提取数据时中使用split函数
        A = ''.join(data_info)
        B = A.split("}")[:-1]

        # 定义gpsData矩阵，shape = (101,6)；存放离散时刻的卫星wgs84坐标下的位置；
        # 6个列元素分别为PX;PY;PZ;VX;VY;VZ;
        # 定义gpsData_time矩阵, shape = (101,1)；存放对应的时刻(timeCode)
        gpsData = np.zeros((len(B), 6))
        gpsData_time = np.zeros((len(B), 1))
        for i in range(0, len(B)):
            tmp = B[i].split(";")
            tmp = tmp[:-1]
            gpsData_time[i] = ''.join(tmp[-8]).split("=")[2]
            for j in range(1, 7):
                gpsData[i][-j] = ''.join(tmp[-j]).split("=")[1]

        return gpsData, gpsData_time

    def load_attData(self,attData_dir):
        """"
        load AttData from DX_ZY3_NAD_att.txt

        input: AddData_dir : file path
        return:
            AttData(array, shape of (groupNumber,4)): q1 q2 q3 q4
            AttData_time(array, shape of(groupNumber,1)): timeCode

        """
        data_info = []  # 存放所有从txt文件中读出的数据
        for line in open(attData_dir, "r"):
            line = line[:-1]  # 逐行读，去掉每行最后的换行符
            data_info.append(line)
        # A为临时变量，将数据格式从list转为string，为了在提取数据时中使用split函数
        A = ''.join(data_info)
        B = A.split("}")[:-1]

        # 定义gpsData矩阵，shape = (401,4) ；存放离散时刻的卫星wgs84坐标下的位置
        # 4个列元素分别为 q1 q2 q3 q4
        # 定义gpsData_time矩阵, shape = (401,1)；存放对应的时刻(timeCode)
        AttData = np.zeros((len(B), 4))
        AttData_time = np.zeros((len(B), 1))
        for i in range(0, len(B)):
            tmp = B[i].split(";")
            tmp = tmp[:-1]
            AttData_time[i] = ''.join(tmp[-12]).split("=")[2]
            for j in range(1, 5):
                AttData[i][-j] = ''.join(tmp[-j]).split("=")[1]

        return AttData, AttData_time

    def load_imagetime(self, imgTime_dir):  # 读取时间文件
        imagetime = np.loadtxt(imgTime_dir, usecols=1, skiprows=1)
        return imagetime

    def transform_time(self,imageTime):
        #basetime
        timecode0 = 131862356.2500000000
        dateTime0 = np.array([[2013,3,7+4.0/24+25/(24*60)+56.25/(24*60*60)]])

        transformed_time=dateTime0.repeat(len(imageTime),axis=0)
        transformed_time[:,-1] = transformed_time[:,-1]+(imageTime - timecode0)/(24*60*60)
        return transformed_time

    def att_interpolate(self,imageTime, AttData, AttData_time):
        """"
            interpolate att
            input:
                imageTime : 成像时刻
                AttData : 存放各离散时刻四元数的矩阵
                AttData_time : 存放与AttData对应的TimeCode
            return:
                qt(array, shape of (N,4)): 每一时刻的四元数
            """
        deltaTime = 0.25
        index = ((imageTime - AttData_time[0]) // deltaTime).astype(np.int)
        t0 = AttData_time[index]
        t = imageTime.reshape(len(imageTime),1)
        t1 = AttData_time[index + 1]

        q0 = AttData[index]
        q1 = AttData[index + 1]
        #统一插值
        theta = np.arccos(np.sum(np.abs(q0*q1),axis=1)).reshape(-1,1)

        # 退化为线性插值情况
        # qt=np.zeros(q0.shape)
        # for ii in range(len(theta)):
        #     d1=t1[ii]-t[ii]
        #     d2=t[ii]-t0[ii]
        #     d =t1[ii]-t0[ii]
        #
        #     if theta[ii]>0.99:
        #         y0=d1/d
        #         y1=d2/d
        #     else:
        #         y0 = np.sin(theta[ii]*d1/d)/np.sin(theta[ii])
        #         y1 = np.sin(theta[ii]*d2/d)/np.sin(theta[ii])
        #     qt[ii] = y0 * q0[ii] + y1 * q1[ii]

        y0 = np.true_divide(np.sin(theta * np.true_divide(t1 - t, t1 - t0)), np.sin(theta))
        y1 = np.true_divide(np.sin(theta * np.true_divide(t - t0, t1 - t0)), np.sin(theta))
        qt = y0 * q0 + y1 * q1

        #绘图测试
        #plt.figure("test_Draw")
        #l=range(len(q0))
        #plt.plot(l,qt[:,0])
        #plt.plot(l,qt[:,1])
        #plt.plot(l,qt[:,2])
        #plt.plot(l,qt[:,3])

        #plt.show()
        # f=open('intermediate_res/qt.txt','w')
        # for i in range(len(q0)):
        #     f.write(str(qt[i,0])+" "+str(qt[i,1])+" "+str(qt[i,2])+" "+str(qt[i,3])+"\n")
        # f.close()

        return qt

    def gps_interpolate(self,imageTime, gpsData, gpsData_time):
        """"
               interpolate P V
               input:
                   imageTime : 成像时刻
                   gpsData : 存放各离散时刻gps数据的矩阵
                   gpsData_time : 存放与gpsData对应的TimeCode
               return:
                   PX(array, shape of (N,3)): 每一时刻的位置向量
                   VX(array, shape of (N,3)): 每一时刻的速度向量
               """
        deltaTime = 1.0
        index = ((imageTime - gpsData_time[0]) // deltaTime).astype(np.int)
        l = len(imageTime)
        tx = imageTime.reshape(l,1)
        t = np.zeros((l,1,9))
        P = np.zeros((l,3,9))
        V = np.zeros((l,3,9))

        for i in range(0, 9):
            t[:,:,i] = gpsData_time[index + i - 4]
            P[:,:,i] = gpsData[index + i - 4, 0:3]
            V[:,:,i] = gpsData[index + i - 4, 3:6]
        PX = VX = np.zeros((l,3))
        for j in range(0, 9):
            s = np.ones((l,1))
            for i in range(0, 9):
                if i == j:
                    continue
                s = np.multiply(s,np.true_divide(tx - t[:,:,i],t[:,:,j] - t[:,:,i]))
            sp = s * P[:,:,j]
            sv = s * V[:,:,j]
            PX = PX + sp
            VX = VX + sv

        return PX, VX



