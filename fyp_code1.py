# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 23:11:35 2020

@author: lavan
"""

import pandas
df=pandas.read_csv(r'cropreq1.csv')
#print(df)
land_info={'texture':('ls','s'),'ph':9,'slope_code':'D','erosion_code':'C','gravel_code':'C',
           'rocky_code':'C','drainage_code':'B','depth_code':'D','temp':30,'rainfall':800,
           'LGP':120}
Ragi={'S1':0,'S2':0,'S3':0,'N':0}
count=0
for i in df.itertuples():
    
    for m,n in land_info.items():
        if n==i.Slope_code:
            #print(n,i.Slope_code)
            count+=1
        if n==i.Erosion_code:
            count+=1
        if n==i.Drainage_code:
            count+=1
        if n==i.Depth_code:
            count+=1
        if n==i.Gravel_code:
            count+=1
        if n==i.Rocky_code:
            count+=1
        if n==i.Temperature:
            count+=1
        if n==i.Rainfall:
            count+=1
        if n==i.LGP:
            count+=1
        if n==i.Texture:
            count+=1
        if n==i.pH:
            count+=1
print(count/30*100,"%")
print("Ragi:Moderately suitable")
            
"""
print(type(land_info))
for i,j in land_info.items():
    print(i,type(i))
    print(j,type(j))
    """
"""
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv(r'cropreq1.csv')
print(dataset)
"""
#dataset.plot(x='Suitability', y='Erosion_code', style='o')  
"""
plt.title('Erosion code vs suitability')  
plt.xlabel('Suitability')  
plt.ylabel('Erosion Code')  
plt.show()"""