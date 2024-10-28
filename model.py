import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 设置中文显示
plt.rcParams['axes.unicode_minus'] = False
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sklearn as sk
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

path1='E:\工作项目\/2024\徐闻项目\其他路段车流量预测\_select_b_cross_name_a_from_keypointcross_vehicle_hour_statistic_202409291005.csv'
path2='E:\工作项目\/2024\徐闻项目\其他路段车流量预测\_select_b_cross_name_a_from_keypointcross_vehicle_hour_statistic_202410111552.csv'

# traffic=pd.DataFrame()
data1 = pd.read_csv(path1,encoding='utf-8')
# road1=湛徐高速k3599+230徐闻往徐闻港方向入港卡口（北往南）
road11=data1[data1['cross_name']=='湛徐高速k3599+230徐闻往徐闻港方向入港卡口（北往南）'].sort_values(by=['day_of_date','hour_of_date'])
road11=road11[['cross_name','day_of_date','hour_of_date','n2s_count']]
road11=road11.loc[road11['day_of_date'] != '2024-09-05' ]
# road2=湛徐高速K3599+231徐闻港往徐闻方向出港卡口（南往北）
road22=data1[data1['cross_name']=='湛徐高速K3599+231徐闻港往徐闻方向出港卡口（南往北）'].sort_values(by=['day_of_date','hour_of_date'])
road22=road22[['cross_name','day_of_date','hour_of_date','s2n_count']]
# road3=徐闻港检查站 出省
road33=data1[data1['cross_name']=='徐闻港检查站 出省'].sort_values(by=['day_of_date','hour_of_date'])
road33=road33[['cross_name','day_of_date','hour_of_date','n2s_count']]
# road4=徐闻港检查站 入省
road44=data1[data1['cross_name']=='徐闻港检查站 入省'].sort_values(by=['day_of_date','hour_of_date'])
road44=road44[['cross_name','day_of_date','hour_of_date','s2n_count']]

data2 = pd.read_csv(path2,encoding='utf-8')
# road1=湛徐高速k3599+230徐闻往徐闻港方向入港卡口（北往南）
road1=data2[data2['cross_name']=='湛徐高速k3599+230徐闻往徐闻港方向入港卡口（北往南）'].sort_values(by=['day_of_date','hour_of_date'])
road1=road1[['cross_name','day_of_date','hour_of_date','n2s_count']]
# road2=湛徐高速K3599+231徐闻港往徐闻方向出港卡口（南往北）
road2=data2[data2['cross_name']=='湛徐高速K3599+231徐闻港往徐闻方向出港卡口（南往北）'].sort_values(by=['day_of_date','hour_of_date'])
road2=road2[['cross_name','day_of_date','hour_of_date','s2n_count']]
# road3=徐闻港检查站 出省
road3=data2[data2['cross_name']=='徐闻港检查站 出省'].sort_values(by=['day_of_date','hour_of_date'])
road3=road3[['cross_name','day_of_date','hour_of_date','n2s_count']]
# road4=徐闻港检查站 入省
road4=data2[data2['cross_name']=='徐闻港检查站 入省'].sort_values(by=['day_of_date','hour_of_date'])
road4=road4[['cross_name','day_of_date','hour_of_date','s2n_count']]

road1=road1.reset_index(drop=True)
road22=road22.reset_index(drop=True)
road33=road33.reset_index(drop=True)
road44=road44.reset_index(drop=True)
# road44=road44.drop('index',axis=1)
def convert_date(date_str):
    # 假设date_str是一个符合日期数字格式的字符串，如'20240302'
    # 使用strptime将其转换为datetime对象
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    # 使用strftime将datetime对象转换为指定的格式，如'%Y-%m-%d'
    formatted_date = date_obj.strftime('%Y-%m-%d')
    return formatted_date

####天气数据清洗
    path_weather='E:\工作项目\/2024\徐闻项目\其他路段车流量预测\ods_live_gdsqxj_202410111614.csv'
    weather= pd.read_csv(path_weather, encoding='utf-8', usecols=['sk_time','sk_p','sk_r1h','sk_wd','sk_wp'])
    weather['sk_time'] = pd.to_datetime(weather['sk_time'])
    weather['sk_time'] = weather['sk_time'].dt.floor('H')  #分钟置为0
    # weather=weather[weather['sk_time']>'2024-09-28 23:00:00']
    weather.rename(columns={'sk_time':'time'},inplace=True)


#车流量数据清洗
def time_revert(data):
    time=[]
    format = "%Y-%m-%d %H:%M"
    for i in range(len(data)):
        if len(str(data['hour_of_date'][i])) < 2:
            data['hour_of_date'][i]= '0' + str(data['hour_of_date'][i])+ ':00'
        else:
            data['hour_of_date'][i]= str(data['hour_of_date'][i])+ ':00'
        # data['hour'] = data.apply(lambda row: str(row['hour_of_date']) + ':00', axis=1)
        string = str(data['day_of_date'][i]) + ' ' + str(data['hour_of_date'][i])
        time_obj = datetime.strptime(string, format)
        time.append(time_obj)
    data.insert(loc=0, column='time', value=time)
    # data.rename(columns={ 'hour_of_date': 'hour'}, inplace=True)
    data.drop(['day_of_date', 'cross_name'], axis=1, inplace=True)
    # 星期编码
    data['day_of_week'] = data['time'].dt.dayofweek  # 星期几
    #小时编码
    le = LabelEncoder()
    data['hour_code'] = le.fit_transform(data['hour_of_date'])
    return data
#节假日标识
holiday = pd.read_excel(r'E:\工作项目\2024\徐闻项目\交通拥堵预测\holiday.xlsx')
Holiday=[]
for i in range(len(traffic)):
    for j in range(len(holiday)):
        if convert_date(traffic['date'][i]) == datetime.strftime(holiday['放假时间'][j],"%Y-%m-%d"):
            hol=1
        else:
            hol=0
    Holiday.append(hol)
road1_new=time_revert(road1)

#两个时间段的数据合并
df_combined = pd.concat([road4_new, road44_new], ignore_index=True)###国庆请的数据和国庆期间数据合并
df_unique= df_combined.drop_duplicates(subset='time', keep='first')##去重
plt.plot(df_unique.time,df_unique['n2s_count'],label='湛徐高速k3599+230徐闻往徐闻港方向入港卡口（北往南）')

merged_df=pd.merge(road1,weather,on='time',how='left')

#获取1-11号的数据建模
df = traffic[~(traffic['date'] == '20240312')]

#添加滞后一阶的标签值数据
X=df[['hour_code','day_of_week','road1', 'road2','road3','capacity']]
y=df['road1'][1:].append(pd.Series([117])).tolist()
X['Y']=y

#数据归一化
# 假设X是包含小时、星期、车流量等特征的数据集
scaler = StandardScaler()  # 或使用MinMaxScaler()
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
mean = scaler.mean_
scale = scaler.scale_

-------------------------------------
#XGBoost模型
import sklearn as sk
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df_scaled[['hour_code','day_of_week','road1', 'road2','road3','capacity']], df_scaled['Y'], test_size=0.2, random_state=42)

# 定义模型的超参数
params = {
   'eta':0.1,
    'max_depth':9,
    'min_child_weight':5,
    'gamma':0.4,
    'subsample':.6,
    'colsample_bytree':.6,
    'reg_alpha':1,
    'objective':'reg:linear'
}

# 将数据转换为DMatrix格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
watchlist1 = [(dtrain,'train'),(dtest,'test')]
# 训练模型
model_xgb = xgb.train(params=params,dtrain=dtrain,num_boost_round=100,early_stopping_rounds=10,evals=watchlist1)
result_xgb = model_xgb.predict(dtest)
mse_xgb = mean_squared_error(y_test,result_xgb)  #0.33

#模型保存到本地
model_xgb.save_model(path+'xgbmodel')
####调用模型
model=xgb.Booster(params)
model.load_model(path+'xgbmodel')

##用模型进行预测
#获取验证数据
traffic_yz=traffic[traffic['date']=='20240312']

# pre_result=[]
# for i in range(len(traffic_yz)):
#     # 新数据归一化
#     yz=pd.DataFrame(traffic_yz[['hour_code','day_of_week','road1', 'road2','road3','capacity']].iloc[i]).transpose()
#     yz_normalized = (yz - mean[:6]) / scale[:6]
#     #预测新数据
#     d_yz = xgb.DMatrix(yz_normalized)
#     result_yz=model_xgb.predict(d_yz)
#     #预测结果逆归一化
#     result_original = result_yz * scale[6:7] + mean[6:7]
#     pre_result.append(result_original[0])

    yz=pd.DataFrame(traffic_yz[['hour_code','day_of_week','road1', 'road2','road3','capacity']].iloc[0]).transpose()
    yz_normalized = (yz - mean[:6]) / scale[:6]
    #预测新数据
    d_yz = xgb.DMatrix(yz_normalized)
    result_yz=model_xgb.predict(d_yz)
    #预测结果逆归一化
    result_original = result_yz * scale[6:7] + mean[6:7]
    pre_result.append(result_original[0])
pd.DataFrame(pre_result).to_csv(path+'3月12号XGB预测结果.xlsx',index=False)


# 显示重要特征
feature_importance = model_xgb.get_score(importance_type='weight')
plot_importance(model_xgb) # road3>road1>hour_code>road2>week
plt.show()

#利用重要程度作为权重，对各个维度的待验证数据的重要程度做排序
I = [a * b for a, b in zip(yz_normalized.iloc[0].tolist(), (49,22,34,16,19,10))]
test=pd.DataFrame(yz_normalized.columns.values.tolist(),columns=['name'],index=None)
test['value']=I
test_desc=test.sort_values(by='value',ascending=False)

#基于sklearn 接口实现xgb
from xgboost import XGBRegressor

model2 = xgb.XGBRegressor(max_depth=9,
                         learning_rate=0.1,
                         n_estimators=100,
                         objective='reg:linear',
                         booster='gbtree',
                         gamma=0.4,
                         min_child_weight=5,
                         subsample=0.6,
                         colsample_bytree=0.6,
                         reg_alpha=0,
                         reg_lambda=1,
                         random_state=0)

watchlist2 = [(X_train,y_train),(X_test,y_test)]
model2.fit(X_train,y_train,eval_set=watchlist2,early_stopping_rounds=10)
result2 = model2.predict(X_test)
mse2 =mean_squared_error(y_test,result2) #1049 r=32

#栅格搜索调参
#调 max_depth,min_child_weight
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
from sklearn.model_selection import GridSearchCV
gsearch =GridSearchCV(estimator=model2,param_grid = param_test1, scoring='neg_mean_squared_error',n_jobs=4, cv=5)
grid_result=gsearch.fit(X_train,y_train)
print("Best: %f using %s" % (grid_result.best_score_,grid_result.best_params_))  #Best: 0.787117 using {'max_depth': 7, 'min_child_weight': 5}
#上下各拓展1
param_test2 = {
 'max_depth':[8,9,10],
 'min_child_weight':[4,5,6]
}
gsearch2 =GridSearchCV(estimator=model2,param_grid = param_test1, scoring='neg_mean_squared_error',n_jobs=4, cv=5)
grid_result2=gsearch2.fit(X_train,y_train)
print("Best: %f using %s" % (grid_result2.best_score_,grid_result2.best_params_)) #Best: 0.787117 using {'max_depth': 7, 'min_child_weight': 5}

#调gamma
param_test3 = {
    'gamma': [i / 10.0 for i in range(0, 5)]
}

gsearch3 =GridSearchCV(estimator=model2,param_grid = param_test3, scoring='neg_mean_squared_error',n_jobs=4, cv=5)
grid_result3=gsearch3.fit(X_train,y_train)
print("Best: %f using %s" % (grid_result3.best_score_,grid_result3.best_params_)) # Best: 0.787750 using {'gamma': 0.1}

#调subsample 和 colsample_bytree参数
param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch4 =GridSearchCV(estimator=model2,param_grid = param_test4, scoring='neg_mean_squared_error',n_jobs=4, cv=5)
grid_result4=gsearch4.fit(X_train,y_train)
print("Best: %f using %s" % (grid_result4.best_score_,grid_result4.best_params_)) # Best: 0.787750 using {'colsample_bytree': 0.6, 'subsample': 0.8}

-----------------------------------------------------------------
#prophet预测
from prophet import Prophet
df_p=pd.DataFrame(df.loc[:,'time'])
Y= df['road1'].tolist()
df_p['y']=Y
df_p=df_p.rename(columns={'time':'ds'})
pro = Prophet(changepoint_prior_scale=0.001, seasonality_prior_scale=0.1)
pro.add_seasonality(name='daily', period=24, fourier_order=5)
model_p=pro.fit(df_p)
future = pro.make_future_dataframe(periods=24,freq='min')
forecast=model_p.predict(future)
fig= model_p.plot(forecast)

forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()

l_pre=forecast['yhat'][:-1].tolist()
df_p['yhat']=l_pre
mse_prophet =mean_squared_error(df_p['y'],df_p['yhat']) #0.21

#用指数平滑法
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 创建 Holt-Winters 季节性模型
model_e = ExponentialSmoothing(y_holt, trend='add', seasonal='add', seasonal_periods=24)
# 拟合模型
model_fit_e = model_e.fit()
# 进行预测
forecast_e = model_fit_e.forecast(steps=24)  # 预测未来24小时的交通流量
pd.DataFrame(forecast_e).to_csv(path+'forecast_e.xlsx',index=False)

---------------------------------------------
##用 岭回归 预测
from sklearn import model_selection as cv
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(df_scaled[['hour_code','day_of_week','road1', 'road2','road3']], df_scaled['Y'], test_size=0.2, random_state=42)
#线性回归
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred)
coef=model.coef_
#岭回归
model = Ridge(alpha=1.0)  # 你可以调整alpha参数
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# 找到最优的alpha
from sklearn.model_selection import cross_val_score
alphas = [0.01, 0.1, 1, 10, 100]
for alpha in alphas:  #  alpha为0.1训练集误差最小为0.2543
    model = Ridge(alpha=alpha)
    scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    print(f'Alpha: {alpha}, Cross-validated MSE: {-scores.mean()}')
---------------------------------------------------------
#支持向量机
import sklearn.svm as svm
svr = svm.SVR(kernel='linear')
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred)   #mse=0.38
# 参数调优
parameters = {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1, 0.5], 'kernel': ['rbf', 'linear']}
grid_search = GridSearchCV(svr, parameters, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
# 最佳参数
print(grid_search.best_params_)
best_svr = grid_search.best_estimator_   #{'C': 0.1, 'epsilon': 0.1, 'kernel': 'linear'}
# 使用最佳参数的模型进行预测
y_pred_best = best_svr.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)   #调参前0.38，后0.50

-----------------------------
#贝叶斯
from sklearn.linear_model import BayesianRidge
# 创建贝叶斯线性回归模型
blr = BayesianRidge()
# 训练模型
blr.fit(X_train, y_train)
# 预测
y_pred_blr = blr.predict(X_test)
# 评估模型
mse_blr= mean_squared_error(y_test, y_pred)   #0.38

-----------------------------------------------------------
#用LSTM模型
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 设置中文显示
plt.rcParams['axes.unicode_minus'] = False
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#max(traffic['road1'])

path='E:\工作项目\/2024\徐闻项目\交通拥堵预测\交通场景文档-双林给的\样例数据\交通卡口、区域、车型流量数据\/410交通数据——20240313\/'

traffic=pd.DataFrame()
name=['正常过车流量k0301.csv','正常过车流量k0302.csv','正常过车流量k0303.csv','正常过车流量k0304.csv','正常过车流量k0305.csv',
      '正常过车流量k0306.csv','正常过车流量k0307.csv','正常过车流量k0308.csv','正常过车流量k0309.csv','正常过车流量k0310.csv',
      '正常过车流量k0311.csv','正常过车流量k0312.csv']
for i in range(len(name)):
    data=pd.read_csv( path+name[i],encoding='gbk',index_col='卡口名称')
    Middle= data[data.index.isin(['\t徐闻环半岛公路入港卡口', '\t徐闻进港大道入港方向卡口'])]
    Middle_transposed = Middle.transpose()
    Middle_transposed.insert(loc=0, column='hour', value=Middle_transposed.index)
    Middle_transposed.insert(loc=0, column='date', value='2024'+name[i][7:11])
    traffic=traffic.append(Middle_transposed)

traffic_gs=pd.DataFrame()

name_gs=['正常过车流量q0301.csv','正常过车流量q0302.csv','正常过车流量q0303.csv','正常过车流量q0304.csv','正常过车流量q0305.csv',
      '正常过车流量q0306.csv','正常过车流量q0307.csv','正常过车流量q0308.csv','正常过车流量q0309.csv','正常过车流量q0310.csv',
      '正常过车流量q0312.csv','正常过车流量q0312.csv']

for i in range(len(name_gs)):
    data=pd.read_csv( path+name_gs[i],encoding='gbk',index_col='区域名称')
    Middle= data[data.index.isin(['\t徐闻县#电警卡口目录#徐闻港收费站出口'])]
    Middle_transposed = Middle.transpose()
    Middle_transposed.insert(loc=0, column='hour', value=Middle_transposed.index)
    Middle_transposed.insert(loc=0, column='date', value='2024'+name[i][7:11])
    traffic_gs=traffic_gs.append(Middle_transposed)
path_LB='E:\工作项目\/2024\徐闻项目\交通拥堵预测\交通场景文档-双林给的\样例数据\港口轮班数据\/3月徐闻港轮班数据\/'

name_LB=['20240301（0-24时）徐闻港区班轮化班期表执行情况统计表','20240302（0-24时）徐闻港区班轮化班期表执行情况统计表',
         '20240303（0-24时）徐闻港区班轮化班期表执行情况统计表','20240304（0-24时）徐闻港区班轮化班期表执行情况统计表',
         '20240305（0-24时）徐闻港区班轮化班期表执行情况统计表','20240306徐闻港区班轮化班期统计报表',
         '20240307徐闻港区班轮化班期统计报表','20240308徐闻港区班轮化班期统计报表','20240309徐闻港区班轮化班期统计报表',
         '20240310徐闻港区班轮化班期统计报表','20240311徐闻港区班轮化班期统计报表','20240312徐闻港区班轮化班期统计报表']

df_lb=pd.DataFrame()
for i in range(len(name_LB)):
    data_LB=pd.read_excel( path_LB+name_LB[i]+'.xls', header=2,usecols=['开航时间','车辆（含大小车）'])
    data_LB['开航时间'] = pd.to_datetime(data_LB['开航时间'])
    data_LB.set_index('开航时间', inplace=True)
    hourly_data = data_LB.resample('H').sum()
    hourly_data.fillna(method='ffill', inplace=True)
    hourly_data.insert(loc=0,column='time',value=hourly_data.index)
    df_lb=df_lb.append(hourly_data)

df_lb = df_lb[~((df_lb['time'] == '20240302 00:00:00') & (df_lb['车辆（含大小车）']== 59) )]#3月1号多出来2号0点的数据，给删掉

def convert_date(date_str):
    # 假设date_str是一个符合日期数字格式的字符串，如'20240302'
    # 使用strptime将其转换为datetime对象
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    # 使用strftime将datetime对象转换为指定的格式，如'%Y-%m-%d'
    formatted_date = date_obj.strftime('%Y-%m-%d')
    return formatted_date

time=[]
format = "%Y-%m-%d %H:%M:%S"
for i in range(len(traffic)):
    string=convert_date(traffic['date'][i])+' '+traffic['hour'][i]+':00'
    time_obj = datetime.strptime(string, format)
    time.append(time_obj)


traffic.insert(loc=0,column='time',value=time)

traffic['road3']=traffic_gs['\t徐闻县#电警卡口目录#徐闻港收费站出口']
traffic.rename(columns={'\t徐闻进港大道入港方向卡口':'road1','\t徐闻环半岛公路入港卡口':'road2'},inplace=True)

#星期编码
traffic['day_of_week'] = traffic['time'].dt.dayofweek  # 星期几
#小时编码
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
traffic['hour_code']= le.fit_transform(traffic['hour'])
#节假日标识
holiday = pd.read_excel(r'E:\工作项目\2024\徐闻项目\交通拥堵预测\holiday.xlsx')
Holiday=[]
for i in range(len(traffic)):
    for j in range(len(holiday)):
        if convert_date(traffic['date'][i]) == datetime.strftime(holiday['放假时间'][j],"%Y-%m-%d"):
            hol=1
        else:
            hol=0
    Holiday.append(hol)

traffic['Holiday']=Holiday
traffic['capacity']=df_lb['车辆（含大小车）'].tolist()

#获取1-11号的数据建模
df = traffic[~(traffic['date'] == '20240312')]

#添加滞后一阶的标签值数据
X=df[['hour_code','day_of_week','road1', 'road2','road3','capacity']]
y=df['road1'][1:].append(pd.Series([117])).tolist()
X['Y']=y

#数据归一化
# 假设X是包含小时、星期、车流量等特征的数据集
scaler = StandardScaler()  # 或使用MinMaxScaler()
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
mean = scaler.mean_
scale = scaler.scale_

#测试集训练集划分
X_train, X_test, y_train, y_test = train_test_split(df_scaled[['hour_code','day_of_week','road1', 'road2','road3','capacity']], df_scaled['Y'], test_size=0.2, random_state=42)

# LSTM [samples, timesteps, features]
#X_train = np.reshape(X_train, (X_train.shape[0], 1,X_train.shape[1]))
X_train = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#创建模型
model = Sequential()
#添加全连接层
model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=128))
#添加输出层
model.add(Dense(units=1, activation='linear'))
#模型编译
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=128)
y_pred = model.predict(X_test)
y_pred_original = y_pred * scale[6:7] + mean[6:7]
y_test_original = y_test * scale[6:7] + mean[6:7]
y_test_original=y_test_original.values.tolist()
y_test=y_test.values.tolist()
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
mse = mean_squared_error(y_test_original, y_pred_original)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))


plt.figure(figsize=(12, 6))
plt.plot(y_pred, label='Predicted')
plt.plot(y_test, label='True')
plt.xlabel('Time Index')
plt.ylabel('Value')
plt.title('Prediction vs True')
plt.legend()
plt.show()

