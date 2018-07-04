import numpy as np
import pandas as pd

arr2 = np.array(np.arange(12)).reshape(4,3)


df1 = pd.DataFrame(arr2)


dic2 = {'a':[1,2,3,4],'b':[5,6,7,8],
'c':[9,10,11,12],'d':[13,14,15,16]}
df2 = pd.DataFrame(dic2)

print(df2)
a=df2.apply( lambda x: min(x) )
print(a)
print(type(a))
df1['add']=df2.apply(lambda x: min(x), axis=1)

df1['aadd']=[10,20,30,40]
print(df1)
print(type(df1))
# print(df2.iloc[:1])
# print(df2[1:2])
# print(df2['a'])
# print(df2[['a']])
