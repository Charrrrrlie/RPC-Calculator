""""
# 摄影时间 Ext.image_time       shape(N,1)
# 儒略历日 Ext.transformed_time shape(N,1)
# 卫星姿态 Ext.att              shape(N,4)
# 卫星位置 Ext.gps_loc          shape(N,3)
# 卫星速度 Ext.gps_v            shape(N,3)

# 相机到本体坐标系 校准矩阵   Rotation.Ru       shape(3,3)
# 本体坐标系到J2000坐标系    Rotation.R_b2j    shape(N_time,3,3)
# J2000坐标系到WGS84坐标系   Rotation.R_j2w   shape(N_time,3,3)
# 初始视向量                Rotation.ux      shape(3,N_pix)
 """""

import numpy as np
import os
import yaml
import time

from multiprocessing.dummy import Pool
from functools import partial
from utility.RPC_utility import RPC
from utility.Ext_utility import Ext
from utility.Rotation_utility import Rotation


if __name__== "__main__":

    count_start_time=time.clock()

    file_path = './data_ZY3'
    inter_path='./intermediate_res'
    out_path='./final'

    if not os.path.exists(inter_path):
        os.makedirs(inter_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

# ————初始值给定—————
    f=open("./yml/params.yml")
    params=yaml.load(f)
    f.close()

    m=params['ctrl_grid_m']  # grid size (w,h)
    n=params['ctrl_grid_n']
    layers=params['ctrl_layers'] # height layers
    z_max=params['max_height']   # height range
    z_min=params['min_height']

    w=params['img_col']
    h=params['img_row']

    iter_times=params['iter_time']  #rpc解算最大循环次数
    iter_thred=params['threshold']  #退出阈值

# ————摄影瞬间对应的外方位元素———————
    attData_dir = file_path+"/DX_ZY3_NAD_att.txt"
    gpsData_dir = file_path+"/DX_ZY3_NAD_gps.txt"
    imagetime_dir = file_path+"/DX_ZY3_NAD_imagingTime.txt"

    exter_element=Ext(attData_dir,gpsData_dir,imagetime_dir)

# ————确定每个像点的旋转矩阵———————
    Direct_dir=file_path+"/NAD.cbr"
    Ru_dir=file_path+"/NAD.txt"
    R_j2w_dir=inter_path+'/rot.txt'
    rot=Rotation(Ru_dir,Direct_dir,R_j2w_dir,exter_element)

    #保存旋转矩阵
    np.save(inter_path+'/j2w.npy',rot.R_j2w)
    np.save(inter_path+'/b2j.npy',rot.R_b2j)
    np.save(inter_path+'/ux.npy',rot.ux)

# ——均匀选取像方坐标  沿扫描线方向排列——————

    # 对每一个扫描行
    photo_loc = [[int((h-1) / n * j),int((w-1) / m * i)] for j in range(0, n+1) for i in range(0, m+1)]
    photo_loc = np.array(photo_loc)

# ————划定格网点————————
    #并行化处理
    def grid_function(range_zip,z0):
        i, j = range_zip   #col,row

        return rot.get_XYZ(rot.ux[:, j], rot.R_b2j[i, :, :], rot.R_j2w[i, :, :]
                                        , exter_element.gps_loc[i, :], z0)

    iter=[[int((h-1) / n * j),int((w-1) / m * i)] for j in range(0, n+1) for i in range(0, m+1)]
    h_layer=[z_min+i*(z_max-z_min)/(layers-1) for i in range(layers)]

    pool = Pool(6)

    ground_points=np.array([])
    count=0
    for ii in h_layer:
        partial_func = partial(grid_function, z0=ii)
        base_points=np.array(pool.map(partial_func,iter)).squeeze()
        if count:
            ground_points=np.concatenate([ground_points,base_points])
        else:
            ground_points=base_points

        count+=1
# ————WGS84 转 BLH——————
    ground_points = rot.XYZ2BLH(ground_points)

    np.save(inter_path+"/gcp.npy",ground_points)
    print("虚拟控制点构建时间：{:.2f} s".format(time.clock()-count_start_time))
    count_pre_time = time.clock()

# ————RPC模型参数求解————————
    photo_loc=np.tile(photo_loc,(layers,1))  #扩展至与gcp相同维数

    rpc=RPC()
    a,b,c,d,condition=rpc.cal(photo_loc,ground_points,iter_times,iter_thred)  #iter_time:20
    #a, b, c, d, condition = rpc.cal_separate(photo_loc, ground_points, 160, 1e-2)  # iter_time:100

    print("RPC解算状态:{}".format(condition))
    print("有理函数多项式求解时间：{:.2f} s".format(time.clock()-count_pre_time))

    if condition:
        rpc.save_res(out_path)   #保存为txt
        rpc.save_npy(out_path)   #保存为npy 便于精度解算
        np.save(inter_path+'/range.npy',np.array([np.max(ground_points[:,0]),np.min(ground_points[:,0]),
                                                        np.max(ground_points[:,1]),np.min(ground_points[:,1])]))
