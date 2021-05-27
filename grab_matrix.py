from selenium import webdriver
from urllib import request
from bs4 import BeautifulSoup
import numpy as np

#获取网站的矩阵为ECEF转ECI 在此场景使用前需要转置（求逆）

path_url="https://hpiers.obspm.fr/eop-pc/index.php?index=matrice_php&lang=en"
#demo_url="http://www.baidu.com"
driver=webdriver.Chrome()
driver.get(path_url)

# name = driver.find_element_by_id("kw")
# name.send_keys("data_operation")

#year an   month mois  day jour
#hour heure  minute min  second sec
name=["an","mois","jour","heure","min","sec"]
t=["2000","06","14","23","59","59"]

path="/html/body/div[7]/center/div/center/form/table/tbody/tr/td[1]/input[5]"

# for i in range(len(name)):
#     time.sleep(2)
#     #info_sender=driver.find_element_by_xpath(path).click()
#     info_sender=driver.find_element_by_name(name[i])
#     info_sender.send_keys(t[i])
#     print(i)

#提交
submit_button = driver.find_element_by_name("SUBMIT")
submit_button.click()

#确认
ensure_button = driver.find_element_by_id("proceed-button")
ensure_button.click()

#获取内容
url_current=driver.current_url
with request.urlopen(url_current) as file:
    data = file.read()
    #print(data)

soup=BeautifulSoup(data,'html.parser')

text=[]
for i in soup.find_all('b'):
    text.append(i.get_text())

matrix=text[-1].split("matrice:")[1].split(" ")

res=[]
for i in matrix:
    if i=='':
        continue
    res.append(i)
res=np.array(res).astype(np.float).reshape(3,3)
print(res)
driver.quit()