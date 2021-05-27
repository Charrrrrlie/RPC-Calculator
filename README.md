# 🛰地形无关控制方案的RPC解算

***卫星摄影测量课程大作业***

利用严格相机模型获得虚拟控制点,通过有理函数模型(RFM)建立物方与像方关系，求解模型参数(RPC).

代码RPC解算部分采用矩阵运算与多线程加速,有较高运行效率.

## Requirements

* Python 3
* MATLAB
* Numpy 1.18.5
* Scipy 1.5.0
* multiprocess 0.70.11.1
* yaml
##
* [GDAL 3.1.4](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal) (Optional)
* OpenCV 4.2.0.34 (Optional)
* matplotlib 2.2.2 (Optional)
* selenium 3.141.0 (Optional)
* beautifulsoup4 4.9.3 (Optional)


##

## Procedure
#### 根据预处理的文件直接解算RPC参数 

结果保存至`final`文件夹中
```
python main.py
```

## **Optional Operation**

#### 外精度评价
根据高精度DEM建立格网,计算外精度.

<small>*(若未安装GDAL,请采用numpy.load方式获取检查点)*</small>
```
python precision_check.py
```



#### 计算J2000至WGS84旋转矩阵 并初始化yaml参数

由python计算扫描行对应`UTC time` 初始化`yml/params.yml` 
```
python preprocessing.py
```
MATLAB`dcmeci2ecef`函数计算旋转矩阵 存储至`intermediate_res/rot.txt`
```
cal_rot_mat.m
```

#### 从[巴黎天文台网站](https://hpiers.obspm.fr/eop-pc/index.php?index=matrice_php&lang=en)获取J2000至WGS84旋转矩阵

当前(2021.5.27)网站输入表单不可用 现提供批量处理Demo. 

<small>ps:网站提供旋转矩阵为WGS84至J2000 当前场景使用时需要转置</small>

```
python grab_matrix.py
```


#### 可视化虚拟格网点及卫星位置结果
```
python plot.py
```

## Things U Should Know

* 有理函数模型方程`易产生病态`,请选择不太小的虚拟控制点层数<small>(eg: k>3)</small>,选择不太严格的迭代收敛方式,并予以初值<small>(类似DLT解算)</small>;
* 膨胀椭球法求交时请选择`离地心点较近的一组解`;
* 可视化结果请使用`BLH大地坐标`进行表达,并将经纬度转换至角度为宜<small>(有时往往不是你算错了而是WGS84坐标系下可视化效果不佳)</small>;
* 使用MATLAB求解求解旋转矩阵时,建议不要使用`ECItoECEF`函数求解,详见[讨论](https://ww2.mathworks.cn/matlabcentral/fileexchange/28233-convert-eci-to-ecef-coordinates);
* 参考文献文件夹`ref`中论文公式`存在部分错误`,仅提供思路参考.

## Acknowledgment
感谢 [zwl-wennin](https://github.com/zwl-wennin) & [Dxy-c](https://github.com/Dxy-c) 对编写工作的贡献.
