import numpy as np
import gdal
import yaml
import cv2
import jdcal
import  os

from utility.Ext_utility import Ext
'''
初始化
获得各扫描行J2000转WGS84坐标旋转矩阵
获得测区高程范围
指定格网点
'''

if __name__ == "__main__":

    ######指定参数##########
    yaml_path='./yml/params.yml'
    yaml_dict={}
    yaml_dict['ctrl_grid_m']=10   # 格网数(m,n) 对应 grid size (w,h)
    yaml_dict['ctrl_grid_n']=10
    yaml_dict['ctrl_layers']=6    # 分层
    yaml_dict['check_grid_m']=20
    yaml_dict['check_grid_n']=20

    yaml_dict['iter_time']=20    #rpc解算最大循环次数
    yaml_dict['threshold']=0.0015 #退出阈值

    img = cv2.imread('./data_ZY3/zy3.tif')
    h, w, _ = img.shape
    yaml_dict['img_row']=h
    yaml_dict['img_col']=w
    ######################

    # ————摄影瞬间对应的外方位元素———————
    file_path = './data_ZY3'
    attData_dir = file_path + "/DX_ZY3_NAD_att.txt"
    gpsData_dir = file_path + "/DX_ZY3_NAD_gps.txt"
    imagetime_dir = file_path + "/DX_ZY3_NAD_imagingTime.txt"

    exter_element = Ext(attData_dir, gpsData_dir, imagetime_dir)

    delta_day = exter_element.transformed_time[:, 2] - (exter_element.transformed_time[:, 2]).astype(np.int)

    # 儒略历日
    # jd_time = []
    # for i in range(len(exter_element.transformed_time)):
    #     jd_time.append(sum(jdcal.gcal2jd(exter_element.transformed_time[i][0], exter_element.transformed_time[i][1],
    #                                      exter_element.transformed_time[i][2])))
    # jd_time = np.array(jd_time)
    # jd_time += delta_day

    output_path="./intermediate_res/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #f=open(output_path+"jd_time.txt","w")

    # for i in jd_time:
    #     f.write(str(i)+'\n')
    # f.close()

    #保存卫星位置
    np.save(output_path + "gps_loc.npy", exter_element.gps_loc)

    #获取高程范围
    f = gdal.Open('./dem/n35_e114_1arc_v3.tif')

    im_width = f.RasterXSize  # 栅格矩阵的列数
    im_height = f.RasterYSize  # 栅格矩阵的行数
    im_data = f.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
    h_max=np.max(im_data)
    h_min=np.min(im_data)
    print("————获取测区DEM信息————")
    print("Height！ max:{}  min:{} ".format(h_max, h_min))

    # 仿射变换得到XY
    transformer = f.GetGeoTransform()

    yaml_dict['max_height']=h_max.item()
    yaml_dict['min_height']=h_min.item()
    f=open(yaml_path,"w")
    yaml.dump(yaml_dict,f)
    f.close()

    def Transform(w, h):
        x = transformer[0] + w * transformer[1] + h * transformer[2]
        y = transformer[3] + w * transformer[4] + h * transformer[5]
        return x, y


    def ReverTrans(x, y):
        mat = np.array([[transformer[1], transformer[2]], [transformer[4], transformer[5]]])
        coe = np.array([[x - transformer[0]], [y - transformer[3]]])
        res = np.dot(np.linalg.inv(mat), coe)
        return res[0], res[1]  # w,h


    top_left_x, top_left_y = Transform(0, 0)
    bottom_right_x, bottom_right_y = Transform(im_width - 1, im_height - 1)

    print("Range! top left:{:.2f} {:.2f}   bottom right:{:.2f} {:.2f}"
          .format(top_left_x, top_left_y, bottom_right_x, bottom_right_y))

    #保存扫描行对应utc时间
    f=open(output_path+"ac_time.txt","w")

    hour=(delta_day*24).astype(np.int)
    minute=((delta_day*24-hour)*60).astype(np.int)
    second=(((delta_day*24-hour)*60)-minute)*60

    for i in range(len(exter_element.transformed_time)):
        f.write(str(int(exter_element.transformed_time[i][0]))+" "+str(int(exter_element.transformed_time[i][1]))+" "
                +str(int(exter_element.transformed_time[i][2]))+" "+str(hour[i])+" "+str(minute[i])+" "+str(second[i])
                +"\n")

    f.close()