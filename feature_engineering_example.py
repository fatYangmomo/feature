import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
import numpy as np

pd.set_option('display.max_columns', 6)


data = pd.read_csv('kaggle_bike_competition_train.csv', header = 0, error_bad_lines=False)


temp = pd.DatetimeIndex(data['datetime'])
data['date'] = temp.date
data['time'] = temp.time


data['hour'] = pd.to_datetime(data.time, format="%H:%M:%S")
data['hour'] = pd.Index(data['hour']).hour


# 我们对时间类的特征做处理，产出一个星期几的类别型变量
data['dayofweek'] = pd.DatetimeIndex(data.date).dayofweek

# 对时间类特征处理，产出一个时间长度变量
data['dateDays'] = (data.date - data.date[0]).astype('timedelta64[D]')

byday = data.groupby('dayofweek')
# 统计下没注册的用户租赁情况
byday['casual'].sum().reset_index()
# 统计下注册的用户的租赁情况
byday['registered'].sum().reset_index()



data['Saturday']=0
data.Saturday[data.dayofweek==5]=1

data['Sunday']=0
data.Sunday[data.dayofweek==6]=1
print(data.head())
print('#################################################')
# remove old data features
dataRel = data.drop(['datetime', 'count','date','time','dayofweek'], axis=1)
#print(dataRel.head())


#我们把连续值的属性放入一个dict中
featureConCols = ['temp','atemp','humidity','windspeed','dateDays','hour']
dataFeatureCon = dataRel[featureConCols]
dataFeatureCon = dataFeatureCon.fillna( 'NA' ) #in case I missed any
X_dictCon = dataFeatureCon.T.to_dict().values()

# 把离散值的属性放到另外一个dict中
featureCatCols = ['season','holiday','workingday','weather','Saturday', 'Sunday']
dataFeatureCat = dataRel[featureCatCols]
dataFeatureCat = dataFeatureCat.fillna( 'NA' ) #in case I missed any
X_dictCat = dataFeatureCat.T.to_dict().values()

# 向量化特征
vec = DictVectorizer(sparse = False)
X_vec_cat = vec.fit_transform(X_dictCat)#向量化后有排序，按a-z顺序
X_vec_con = vec.fit_transform(X_dictCon)

print(dataFeatureCon.head())
print(X_vec_con)

print(dataFeatureCat.head())
print(X_vec_cat)

#对连续特征,标准化，不标准化就给训练，梯度下降非常困难。因为偏导有的大有的小，下降时震荡形式。则收敛慢，准确度低。
scaler = preprocessing.StandardScaler().fit(X_vec_con)
X_vec_con = scaler.transform(X_vec_con)

#对类别特征进行one-hot编码
enc = preprocessing.OneHotEncoder()
enc.fit(X_vec_cat)
X_vec_cat = enc.transform(X_vec_cat).toarray()

X_vec = np.concatenate((X_vec_con,X_vec_cat), axis=1)

# 对结果Y向量化（租车数量）
Y_vec_reg = dataRel['registered'].values.astype(float)
Y_vec_cas = dataRel['casual'].values.astype(float)