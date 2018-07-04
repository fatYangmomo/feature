# pandas and numpy for data manipulation
import pandas as pd
from pandas import merge
import numpy as np

# featuretools for automated feature engineering
import featuretools as ft

# ignore warnings from pandas
import warnings
warnings.filterwarnings('ignore')
# Read in the data,读取后为Dataframe格式
clients = pd.read_csv('data/clients.csv', parse_dates = ['joined'])
loans = pd.read_csv('data/loans.csv', parse_dates = ['loan_start', 'loan_end'])
payments = pd.read_csv('data/payments.csv', parse_dates = ['payment_date'])

print(   clients.head()   )

pd.set_option('display.max_columns',10)

# Create a month column
clients['join_month'] = clients['joined'].dt.month

# Create a log of income column
clients['log_income'] = np.log(clients['income'])

#print(   clients.head()   )
#groupby根据client_id进行分组聚合，单对['loan_amount']列聚合，agg每次传入的是一列数据，对其聚合后返回标量。
stats = loans.groupby('client_id')['loan_amount'].agg(['mean', 'max', 'min'])
stats.columns = ['mean_loan_amount', 'max_loan_amount', 'min_loan_amount']
print( stats.head() )
#left_on：左则DataFrame中用作连接键的列名;这个参数中左右列名不相同，但代表的含义相同时非常有用。
#how：指的是合并(连接)的方式有inner(内连接),left(左外连接),right(右外连接),outer(全外连接);默认为inner
#right_index：使用右则DataFrame中的行索引做为连接键
a=clients.merge(stats, left_on = 'client_id', right_index=True, how = 'left').head(10)
#等同于xx=merge(clients,stats,on='client_id',how='left')
print( a.head() )

#we initialize an EntitySet and give it an id
es = ft.EntitySet(id = 'clients')

#加载dataframe作为实体,index参数指定唯一标识数据帧中的行的列,即索引；time_index说明数据何时创建？；
# variable_types参数指示“repaid”应该被解释为一个Categorical变量，尽管它只是底层数据中的一个整数。
es = es.entity_from_dataframe(entity_id = 'clients', dataframe = clients,
                              index = 'client_id', time_index = 'joined')

es = es.entity_from_dataframe(entity_id = 'loans', dataframe = loans,
                              variable_types = {'repaid': ft.variable_types.Categorical},
                              index = 'loan_id',
                              time_index = 'loan_start')

es = es.entity_from_dataframe(entity_id = 'payments',
                              dataframe = payments,
                              variable_types = {'missed': ft.variable_types.Categorical},
                              make_index = True,
                              index = 'payment_id',
                              time_index = 'payment_date')

# Relationship between clients and previous loans
r_client_previous = ft.Relationship(es['clients']['client_id'],
                                    es['loans']['client_id'])

# Add the relationship to the entity set
es = es.add_relationship(r_client_previous)


r_payments = ft.Relationship(es['loans']['loan_id'],
                                      es['payments']['loan_id'])

# Add the relationship to the entity set
es = es.add_relationship(r_payments)

print(es)


primitives = ft.list_primitives()
pd.options.display.max_colwidth = 100
b=primitives[primitives['type'] == 'aggregation'].head(10)
print(b)
#默认max_depth为2？
print(es)
print(clients)
features, feature_names = ft.dfs(entityset = es, target_entity = 'clients',
                                 agg_primitives = ['mean', 'max', 'percent_true', 'last'],
                                 trans_primitives = ['years', 'month', 'subtract', 'divide']
                                 )
print(es)
print(clients)
print( features.head() )
print( feature_names )
c=pd.DataFrame(features['MONTH(joined)'].head())
print(c)
d=pd.DataFrame(features['MEAN(payments.payment_amount)'].head())
print(d)


pd.DataFrame(features['MEAN(loans.loan_amount)'].head(10))

pd.DataFrame(features['LAST(loans.MEAN(payments.payment_amount))'].head(10))

print( features.head() )
features, feature_names = ft.dfs(entityset=es, target_entity='clients',
                                 max_depth = 2)
print( features.head() )
print( feature_names )
print( features.iloc[:, 4:].head() )