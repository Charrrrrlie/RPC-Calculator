import numpy as np
import gdal
import yaml
from utility.RPC_utility import RPC


def BLH2XYZ(BLH):
    a = 6378137.0000
    b = 6356752.3141
    pi = 3.1415926
    e2 = 1 - (b / a) ** 2

    B = BLH[:, 0]
    L = BLH[:, 1]
    H = BLH[:, 2]

    L = L * pi / 180
    B = B * pi / 180

    N = a / np.sqrt(1 - e2 * np.sin(B) ** 2)
    x = (N + H) * np.cos(B) * np.cos(L)
    y = (N + H) * np.cos(B) * np.sin(L)
    z = (N * (1 - e2) + H) * np.sin(B)
    return np.vstack((x, y, z)).transpose()

def XYZ2BLH(XYZ):

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

def Transform(transformer,w,h):
    x=transformer[0]+w*transformer[1]+h*transformer[2]
    y=transformer[3]+w*transformer[4]+h*transformer[5]
    return x,y

def ReverTrans(transformer,x,y):
    mat=np.array([[transformer[1],transformer[2]],[transformer[4],transformer[5]]])
    coe=np.array([[x-transformer[0]],[y-transformer[3]]])
    res=np.dot(np.linalg.inv(mat),coe)
    return res[0],res[1]  # w,h

def GetCheckGrid(path_range,path_dem,m,n):

    f = gdal.Open(path_dem)

    im_width = f.RasterXSize  # 栅格矩阵的列数
    im_height = f.RasterYSize  # 栅格矩阵的行数
    im_data = f.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
    transformer = f.GetGeoTransform()

    # Range
    #35.9, 35.84 114.8, 114.69
    y_max, y_min, x_max, x_min = np.load(path_range)

    y_max-=0.05
    y_min+=0.05
    x_max-=0.05
    x_min+=0.05

    w_max, h_max = ReverTrans(transformer,x_max, y_min)
    w_min, h_min = ReverTrans(transformer,x_min, y_max)

    # 均分格网点
    w, h = w_max, h_max
    dx, dy = w_min, h_min

    pts = [[int((h - dy) / n * j + dy), int((w - dx) / m * i + dx)] for j in range(0, n + 1) for i in range(0, m + 1)]

    res = []
    for pt in pts:
        l, b = Transform(transformer,pt[1], pt[0])
        hh = im_data[pt[0], pt[1]]
        res.append([b, l, hh])

    return np.array(res)

# ————外精度检查——————————
if __name__== "__main__":

    root_path='./intermediate_res/'

    gps_loc=np.load(root_path+'gps_loc.npy')
    R_j2w=np.load(root_path+'j2w.npy')
    R_b2j=np.load(root_path+'b2j.npy')
    ux=np.load(root_path+'ux.npy')

    #check_pt=np.load(root_path+"checkpt.npy")

    #根据格网获取检查点
    f = open("./yml/params.yml")
    params = yaml.load(f)
    f.close()
    grid_m=params['check_grid_m']
    grid_n=params['check_grid_n']
    check_pt = GetCheckGrid(root_path + 'range.npy','./dem/n35_e114_1arc_v3.tif',grid_m,grid_n)
    np.save(root_path+"checkpt.npy",check_pt)

    coeff=np.load('./final/coeff.npy')
    partial=np.load('./final/partial.npy')
    rpc=RPC()
    rpc.init_from_outer(coeff,partial)

    check_pt = BLH2XYZ(check_pt)
    pho_check, ground_check = rpc.check(check_pt, gps_loc, R_j2w, R_b2j,ux)

    ground_check = XYZ2BLH(ground_check)
    pho_check, ground_check = rpc.norm_gravi(pho_check, ground_check, rpc.center)
    l = pho_check.shape[0]

    # 计算像点坐标
    pho_check_pred = rpc.cal_pho_from_RFM(ground_check)

    print("——精度评价结果———")
    # 计算去重心化的坐标为宜
    # error_row = np.mean(np.abs(pho_check_pred[:, 0] - pho_check[:, 0]))
    # error_col = np.mean(np.abs(pho_check_pred[:, 1] - pho_check[:, 1]))
    # print('MAE in ROW(normalized):{}'.format(error_row))
    # print('MAE in COL(normalized):{}'.format(error_col))

    # 去重心化
    pho_check_pred = rpc.de_norm_gravi_pho(pho_check_pred, rpc.center)
    pho_check = rpc.de_norm_gravi_pho(pho_check, rpc.center)

    # MAE -mean absolute error
    error_row = np.mean(np.abs(pho_check_pred[:, 0] - pho_check[:, 0]))
    error_col = np.mean(np.abs(pho_check_pred[:, 1] - pho_check[:, 1]))

    print('MAE in ROW:{}'.format(error_row))
    print('MAE in COL:{}'.format(error_col))
