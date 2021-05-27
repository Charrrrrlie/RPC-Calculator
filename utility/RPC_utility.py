import numpy as np
import sys
"""
rn=p1(X,Y,Z)/p2(X,Y,Z)
cn=p3(X,Y,Z)/p4(X,Y,Z)
像方坐标 pho   重心化后 pho_g
物方坐标 obj   重心化后 obj_g
"""

class RPC:

    def __init__(self):
        self.a=np.ones(20)
        self.b=np.ones(20)
        self.c=np.ones(20)
        self.d=np.ones(20)

    #重心化 返回像方col row物方 XYZ 重心
    def gravity(self,pho,obj):

        #gravity center
        c_g=pho[:,1].flatten().mean()
        r_g=pho[:,0].flatten().mean()

        c_s = max(np.max(pho[:,1]) - c_g, c_g - np.min(pho[:,1]))
        r_s = max(np.max(pho[:,0]) - r_g, r_g - np.min(pho[:,0]))

        X_g=obj[:,0].flatten().mean()
        Y_g=obj[:,1].flatten().mean()
        Z_g=obj[:,2].flatten().mean()

        X_s= max(np.max(obj[:,0])-X_g,X_g-np.min(obj[:,0]))
        Y_s= max(np.max(obj[:,1])-Y_g,Y_g-np.min(obj[:,1]))
        Z_s= max(np.max(obj[:,2])-Z_g,Z_g-np.min(obj[:,2]))


        return [c_g,r_g,c_s,r_s, X_g,Y_g,Z_g,X_s,Y_s,Z_s]

    #行列一同解算 以向量形式计算
    def diff(self,pho_loc,obj_loc):

        #column vector
        #物方坐标
        X=obj_loc[:,0].reshape(-1,1)
        Y=obj_loc[:,1].reshape(-1,1)
        Z=obj_loc[:,2].reshape(-1,1)
        #像方坐标
        R=pho_loc[:,0].reshape(-1,1)
        C=pho_loc[:,1].reshape(-1,1)

        l=len(X)

        vec=np.hstack([np.ones((l,1)),Z,Y,X,Z*Y,Z*X,Y*X,Z*Z,Y*Y,X*X,Z*Y*X,Z*Z*Y,Z*Z*X
                            ,Y*Y*Z,Y*Y*X,Z*X*X,Y*X*X,Z*Z*Z,Y*Y*Y,X*X*X])

        # 误差方程系数  row
        # J = np.concatenate([self.a, self.b[1:]]).transpose()
        B=np.dot(vec,self.b.reshape(-1,1))
        M=np.concatenate([vec, -np.multiply(R,vec[:,1:])], axis=1)
        W_R=np.true_divide(np.identity(l),B)

        #column
        # K = np.concatenate([self.c, self.d[1:]]).transpose()
        D = np.dot(vec, self.d.reshape(-1,1))
        N=np.concatenate([vec,-np.multiply(C,vec[:,1:])],axis=1)
        W_C = np.true_divide(np.identity(l),D)

        # 法方程
        # AT*W*W*A * U= AT*W*W *L
        # U=np.concatenate([J,K])
        A=np.block(
            [
                [M    ,   np.zeros(N.shape)],
                [np.zeros(M.shape), N]
            ])
        W=np.block(
            [
                [W_R, np.zeros(W_C.shape)],
                [np.zeros(W_R.shape), W_C]
            ]
        )
        L=np.concatenate([R,C])

        mat_t=np.dot(np.dot(np.dot(A.transpose(),W),W),A)
        det_val=np.linalg.det(mat_t)
        if det_val<1e-2:
            print("接近奇异矩阵！特征值为：{}".format(det_val))

        U=np.dot(np.dot(np.dot(np.dot(np.linalg.inv(mat_t),A.transpose()),W),W),L)
        v=np.dot(np.dot(W,A),U)-np.dot(W,L)
        return U,v

    #行列分别解算
    def diff_separate(self,pho_loc,obj_loc):
        # column vector
        # 物方坐标
        X = obj_loc[:, 0].reshape(-1, 1)
        Y = obj_loc[:, 1].reshape(-1, 1)
        Z = obj_loc[:, 2].reshape(-1, 1)
        # 像方坐标
        R = pho_loc[:, 0].reshape(-1, 1)
        C = pho_loc[:, 1].reshape(-1, 1)

        l = len(X)

        vec = np.hstack(
            [np.ones((l, 1)), Z, Y, X, Z * Y, Z * X, Y * X, Z * Z, Y * Y, X * X, Z * Y * X, Z * Z * Y, Z * Z * X
                , Y * Y * Z, Y * Y * X, Z * X * X, Y * X * X, Z * Z * Z, Y * Y * Y, X * X * X])

        # 误差方程系数  row
        # J = np.concatenate([self.a, self.b[1:]]).transpose()
        B = np.dot(vec, self.b.reshape(-1, 1))
        M = np.concatenate([vec, -np.multiply(R, vec[:, 1:])], axis=1)
        W_R = np.true_divide(np.identity(l), B)

        # column
        # K = np.concatenate([self.c, self.d[1:]]).transpose()
        D = np.dot(vec, self.d.reshape(-1, 1))
        N = np.concatenate([vec, -np.multiply(C, vec[:, 1:])], axis=1)
        W_C = np.true_divide(np.identity(l), D)

        mat_row = np.dot(np.dot(np.dot(M.transpose(), W_R), W_R), M)
        mat_col = np.dot(np.dot(np.dot(N.transpose(), W_C), W_C), N)

        J = np.dot(np.dot(np.dot(np.dot(np.linalg.inv(mat_row), M.transpose()), W_R), W_R), R)
        K = np.dot(np.dot(np.dot(np.dot(np.linalg.inv(mat_col), N.transpose()), W_C), W_C), C)
        v_R = np.dot(np.dot(W_R, M), J) - np.dot(W_R, R)
        v_C = np.dot(np.dot(W_C, N), K) - np.dot(W_C, C)
        return J, K, v_R, v_C

    #初始化
    def initialize(self,pho_loc,obj_loc):
        # column vector
        # 物方坐标
        X = obj_loc[:, 0].reshape(-1, 1)
        Y = obj_loc[:, 1].reshape(-1, 1)
        Z = obj_loc[:, 2].reshape(-1, 1)
        # 像方坐标
        R = pho_loc[:, 0].reshape(-1, 1)
        C = pho_loc[:, 1].reshape(-1, 1)

        l = len(X)

        vec = np.hstack(
            [np.ones((l, 1)), Z, Y, X, Z * Y, Z * X, Y * X, Z * Z, Y * Y, X * X, Z * Y * X, Z * Z * Y, Z * Z * X
                , Y * Y * Z, Y * Y * X, Z * X * X, Y * X * X, Z * Z * Z, Y * Y * Y, X * X * X])
        M=np.concatenate([vec, -np.multiply(R, vec[:, 1:])], axis=1)
        N=np.concatenate([vec, -np.multiply(C, vec[:, 1:])], axis=1)

        A = np.block(
            [
                [M, np.zeros(N.shape)],
                [np.zeros(M.shape), N]
            ])
        L = np.concatenate([R, C])

        temp_init=np.dot(np.dot(np.linalg.inv(np.dot(A.transpose(),A)),A.transpose()),L)
        self.a=temp_init[0:20].flatten(order='F')
        self.b[1:]=temp_init[20:39].flatten(order='F')
        self.c=temp_init[39:59].flatten(order='F')
        self.d[1:]=temp_init[59:78].flatten(order='F')

        #temp_init

    #行列分别解算
    def initialize_separate(self,pho_loc,obj_loc):
        # column vector
        # 物方坐标
        X = obj_loc[:, 0].reshape(-1, 1)
        Y = obj_loc[:, 1].reshape(-1, 1)
        Z = obj_loc[:, 2].reshape(-1, 1)
        # 像方坐标
        R = pho_loc[:, 0].reshape(-1, 1)
        C = pho_loc[:, 1].reshape(-1, 1)

        l = len(X)

        vec = np.hstack(
            [np.ones((l, 1)), Z, Y, X, Z * Y, Z * X, Y * X, Z * Z, Y * Y, X * X, Z * Y * X, Z * Z * Y, Z * Z * X
                , Y * Y * Z, Y * Y * X, Z * X * X, Y * X * X, Z * Z * Z, Y * Y * Y, X * X * X])
        M = np.concatenate([vec, -np.multiply(R, vec[:, 1:])], axis=1)
        N = np.concatenate([vec, -np.multiply(C, vec[:, 1:])], axis=1)

        ab_init = np.dot(np.dot(np.linalg.inv(np.dot(M.transpose(), M)), M.transpose()), R)
        cd_init = np.dot(np.dot(np.linalg.inv(np.dot(N.transpose(), N)), N.transpose()), C)
        self.a = ab_init[0:20].flatten(order='F')
        self.b[1:] = ab_init[20:39].flatten(order='F')
        self.c = cd_init[0:20].flatten(order='F')
        self.d[1:] = cd_init[20:39].flatten(order='F')

    #迭代求解系数
    def cal(self,pho,obj,times,thred):

        #重心化 c_g,r_g,c_s,r_s, X_g,Y_g,Z_g,X_s,Y_s,Z_s
        center_g=self.gravity(pho,obj)

        self.center=center_g

        pho_n=pho.astype(np.float)
        obj_n=obj.astype(np.float)
        pho_n[:,1] = (pho_n[:,1]-center_g[0])/center_g[2] # col
        pho_n[:,0] = (pho_n[:,0]-center_g[1])/center_g[3] # row

        obj_n[:,0] = (obj_n[:,0]-center_g[4])/center_g[7] # B
        obj_n[:,1] = (obj_n[:,1]-center_g[5])/center_g[8] # L
        obj_n[:,2] = (obj_n[:,2]-center_g[6])/center_g[9] # H

        #初始化！
        self.initialize(pho_n,obj_n)

        count=1
        #循环迭代
        while(times):

            temp_u,v=self.diff(pho_n,obj_n)
            temp_a=temp_u[0:20].flatten(order='F')
            temp_b=temp_u[20:39].flatten(order='F')   #b从b1开始
            temp_c=temp_u[39:59].flatten(order='F')
            temp_d=temp_u[59:78].flatten(order='F')   #d从d1开始

            # print(np.max(abs(self.a-temp_a)))
            # print(np.max(abs(self.b[1:]-temp_b)))
            # print(np.max(abs(self.c-temp_c)))
            # print(np.max(abs(self.d[1:]-temp_d)))
            # print('——————————————————')


            #if (abs(self.a-temp_a)<thred).all() and (abs(self.b[1:]-temp_b)<thred).all() \
             #   and (abs(self.c-temp_c)<thred).all() and (abs(self.d[1:]-temp_d)<thred).all() :

              #  return temp_a,temp_b,temp_c,temp_d,True
            #print(np.max(abs(v)))

            if(np.max(abs(v))<thred):
                #print(temp_a, temp_b, temp_c, temp_d)
                print('迭代次数:{}'.format(count))
                return temp_a, temp_b, temp_c, temp_d, True
            # update
            else:
                self.a=temp_a
                self.b[1:]=temp_b
                self.c=temp_c
                self.d[1:]=temp_d
            times-=1
            count+=1

        negative=np.array([])
        return negative,negative,negative,negative,False

    # 行列分别解算
    def cal_separate(self,pho,obj,times,thred):

        #重心化 c_g,r_g,c_s,r_s, X_g,Y_g,Z_g,X_s,Y_s,Z_s
        center_g=self.gravity(pho,obj)

        self.center=center_g

        pho_n=pho.astype(np.float)
        obj_n=obj.astype(np.float)
        pho_n[:,1] = (pho_n[:,1]-center_g[0])/center_g[2] # col
        pho_n[:,0] = (pho_n[:,0]-center_g[1])/center_g[3] # row

        obj_n[:,0] = (obj_n[:,0]-center_g[4])/center_g[7] # B
        obj_n[:,1] = (obj_n[:,1]-center_g[5])/center_g[8] # L
        obj_n[:,2] = (obj_n[:,2]-center_g[6])/center_g[9] # H

        #初始化！
        self.initialize_separate(pho_n,obj_n)

        count=1
        #循环迭代
        while(times):

            J,K,J_v,K_v=self.diff_separate(pho_n,obj_n)

            temp_a=J[0:20].flatten(order='F')
            temp_b=J[20:39].flatten(order='F')   #b从b1开始
            temp_c=K[0:20].flatten(order='F')
            temp_d=K[20:39].flatten(order='F')   #d从d1开始

            print(np.max(abs(self.a-temp_a)))
            print(np.max(abs(self.b[1:]-temp_b)))
            print(np.max(abs(self.c-temp_c)))
            print(np.max(abs(self.d[1:]-temp_d)))
            print('——————————————————')


            if (abs(self.a-temp_a)<thred).all() and (abs(self.b[1:]-temp_b)<thred).all() \
               and (abs(self.c-temp_c)<thred).all() and (abs(self.d[1:]-temp_d)<thred).all() :

               return temp_a,temp_b,temp_c,temp_d,True
            #print(np.max(abs(v)))

            # if(np.max(abs(v))<thred):
            #     #print(temp_a, temp_b, temp_c, temp_d)
            #     print('迭代次数:{}'.format(count))
            #     return temp_a, temp_b, temp_c, temp_d, True
            # update
            else:
                self.a=temp_a
                self.b[1:]=temp_b
                self.c=temp_c
                self.d[1:]=temp_d
            times-=1
            count+=1

        negative=np.array([])
        return negative,negative,negative,negative,False

    def save_res(self, path):
        f = open(path+'/rpc_results.txt', "w")

        ncenter=np.ones(10)
        ncenter[0]=self.center[1]
        ncenter[1] = self.center[0]
        ncenter[2] = self.center[4]
        ncenter[3] = self.center[5]
        ncenter[4] = self.center[6]
        ncenter[5] = self.center[3]
        ncenter[6] = self.center[2]
        ncenter[7] = self.center[7]
        ncenter[8] = self.center[8]
        ncenter[9] = self.center[9]
        # 先存偏置
        list=['LINE_OFF: ','SAMP_OFF: ','LAT_OFF: ','LONG_OFF: ','HEIGHT_OFF: ','LINE_SCALE: ','SAMP_SCALE: ','LAT_SCALE: ','LONG_SCALE: ','HEIGHT_SCALE: ']
        list1=[' pixels',' pixels',' degrees',' degrees',' meters',' pixels',' pixels',' degrees',' degrees',' meters']
        for i in range(10):
            f.write(list[i]+str(ncenter[i])+list1[i]+"\n")

        #rpc78参数存储
        list2 = ['LINE_NUM_COEFF_', 'LINE_DEN_COEFF_', 'SAMP_NUM_COEFF_', 'SAMP_DEN_COEFF_']
        for i in range(20):
            f.write(list2[0] + str(i + 1) + ":    " + str(self.a[i]) + "\n")
        for i in range(20):
            f.write(list2[1] + str(i + 1) + ":    " + str(self.b[i]) + "\n")
        for i in range(20):
            f.write(list2[2] + str(i + 1) + ":    " + str(self.c[i]) + "\n")
        for i in range(20):
            f.write(list2[3] + str(i + 1) + ":    " + str(self.d[i]) + "\n")

        f.close()

    def save_npy(self,path):
        coeff_path=path+'/coeff.npy'
        patial_path=path+'/partial.npy'

        np.save(patial_path,self.center)
        self.a=self.a.reshape(-1,1)
        self.b=self.b.reshape(-1,1)
        self.c=self.c.reshape(-1,1)
        self.d=self.d.reshape(-1,1)

        coeff=np.hstack((self.a,self.b,self.c,self.d))
        np.save(coeff_path,coeff)

    #窗口二分法求外精度
    def check(self,data,gps_loc,R_j2w,R_b2j,ux):

        check_x = []
        check_y = []

        #无效值
        invalid=[]

        line_num=gps_loc.shape[0]

        for i in range(len(data)):

            XYZ=data[i,:]
            Ns = 0
            Ne = line_num - 1
            iteration = True

            while iteration:

                N_ = int((Ns + Ne) / 2)

                #calculate pixel loc
                xs, ys = self.cal_phopt(Ns,R_j2w,R_b2j, XYZ, gps_loc)
                x_, y_ = self.cal_phopt(N_,R_j2w,R_b2j, XYZ, gps_loc)
                xe, ye = self.cal_phopt(Ne,R_j2w,R_b2j, XYZ, gps_loc)

                if xs < 0 and x_ < 0 and xe < 0:
                    iteration = False
                    invalid.append(i)
                elif xs > 0 and x_ > 0 and xe > 0:
                    iteration = False
                    invalid.append(i)
                else:
                    if (xs * x_ <= 0):
                        Ne = N_
                    elif (x_ * xe <= 0):
                        Ns = N_
                    else:
                        Ns = int((Ns + N_) / 2)
                        Ne = int((Ne + N_) / 2)

                    if abs(Ne - Ns) <= 1:
                        iteration = False
                        check_x.append(int(N_))
                        check_y.append(int(np.argmin(np.abs(y_-ux[1,:]))))
        #row col
        return np.vstack((check_x, check_y)).transpose().astype(np.float), np.delete(data,invalid,axis=0)

    #共线条件方程计算严密像点坐标
    def cal_phopt(self,N,R_j2w,R_b2j,XYZ,gps_loc):
        XYZs = gps_loc[N]
        R=np.dot(R_j2w[N],R_b2j[N])
        R=np.linalg.inv(R)
        XYZ_hat=np.dot(R,XYZ-XYZs)
        return -XYZ_hat[0]/XYZ_hat[2],-XYZ_hat[1]/XYZ_hat[2]

    #以rpc相同偏置归一
    def norm_gravi(self,pho,obj,center):
        # 重心化 c_g,r_g,c_s,r_s, X_g,Y_g,Z_g,X_s,Y_s,Z_s
        pho[:, 1] = (pho[:, 1] - center[0]) / center[2]  # col
        pho[:, 0] = (pho[:, 0] - center[1]) / center[3]  # row

        obj[:, 0] = (obj[:, 0] - center[4]) / center[7]  # B
        obj[:, 1] = (obj[:, 1] - center[5]) / center[8]  # L
        obj[:, 2] = (obj[:, 2] - center[6]) / center[9]  # H

        return pho,obj

    #去归一化
    def de_norm_gravi_pho(self,pho,center):
        pho[:,1]=pho[:, 1]*center[2]+center[0]     #col
        pho[:,0]=pho[:,0]*center[3]+center[1]      #row
        return pho

    def de_norm_gravi_obj(self, obj, center):
        obj[:, 0] = obj[:, 0]*center[7] + center[4]     # B
        obj[:, 1] = obj[:, 1]*center[8] + center[5]    # L
        obj[:, 2] = obj[:, 2]*center[9] + center[6]    # H
        return obj

    #利用正解变换计算像点坐标
    def cal_pho_from_RFM(self,obj):
        # 物方坐标
        X = obj[:, 0].reshape(-1, 1)
        Y = obj[:, 1].reshape(-1, 1)
        Z = obj[:, 2].reshape(-1, 1)
        l=len(X)

        vec = np.hstack(
            [np.ones((l, 1)), Z, Y, X, Z * Y, Z * X, Y * X, Z * Z, Y * Y, X * X, Z * Y * X, Z * Z * Y, Z * Z * X
                , Y * Y * Z, Y * Y * X, Z * X * X, Y * X * X, Z * Z * Z, Y * Y * Y, X * X * X])

        row=np.true_divide(np.dot(vec,self.a),np.dot(vec,self.b))
        col=np.true_divide(np.dot(vec,self.c),np.dot(vec,self.d))

        return np.vstack((row,col)).transpose()

    #外源初始化
    def init_from_outer(self,coeff,center):
        self.center=center
        self.a=coeff[:,0]
        self.b=coeff[:,1]
        self.c=coeff[:,2]
        self.d=coeff[:,3]