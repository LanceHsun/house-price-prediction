# 1、导入所需的库
import numpy as np

import pandas as pd
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))

import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

import matplotlib.pyplot as plt

from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import norm, skew

#忽略警告
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------
# 2、查看数据集大小

train = pd.read_csv('train.csv')
# print('The shape of training data:', train.shape)
# print(train.head())

test = pd.read_csv('test.csv')
# print('The shape of testing data:', test.shape)
# print(test.head())

# -----------------------------------------------------------------

#ID列没有用，直接删掉
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
# print('The shape of training data:', train.shape)
# print('The shape of testing data:', test.shape)

# -----------------------------------------------------------------
# 3、目标值分析

#分离数字特征和类别特征
num_features = []
cate_features = []
for col in test.columns:
    if test[col].dtype == 'object':
        cate_features.append(col)
    else:
        num_features.append(col)
# print('number of numeric features:', len(num_features))
# print('number of categorical features:', len(cate_features))

# -----------------------------------------------------------------
# 5、异常值处理

#处理掉x='TotalBsmtSF', y='SalePrice'右下的明显异常值
train = train.drop(train[(train['TotalBsmtSF']>6000) & (train['SalePrice']<200000)].index)
# # 绘制 TotalBsmtSF 和 SalePrice 的散点图，确认异常值是否已删除
# sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=train)
# plt.show()  # 显示图像


#处理掉x='GrLivArea', y='SalePrice'右下的异常值
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<200000)].index)
# # 绘制 TotalBsmtSF 和 SalePrice 的散点图，确认异常值是否已删除
# sns.scatterplot(x='GrLivArea', y='SalePrice', data=train)
# plt.show()  # 显示图像

# -----------------------------------------------------------------
# 6、缺失值处理

# # 查看训练集中各特征的数据缺失个数
# print('The shape of training data:', train.shape)
# train_missing = train.isnull().sum()
# train_missing = train_missing.drop(train_missing[train_missing==0].index).sort_values(ascending=False)
# train_missing

# #查看测试集中各特征的数据缺失个数
# print('The shape of testing data:', test.shape)
# test_missing = test.isnull().sum()
# test_missing = test_missing.drop(test_missing[test_missing==0].index).sort_values(ascending=False)
# test_missing

# -----------------------------------------------------------------

# 策略一：类别特征补'None'
none_lists = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinType1',
              'BsmtFinType2', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'MasVnrType']
for col in none_lists:
    train[col] = train[col].fillna('None')
    test[col] = test[col].fillna('None')

# 策略二：类别特征非‘None’的补充出现最多的类别
most_lists = ['MSZoning', 'Exterior1st', 'Exterior2nd', 'SaleType', 'KitchenQual', 'Electrical']
for col in most_lists:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(train[col].mode()[0])    #注意这里补充的是训练集中出现最多的类别
# 特例：
# 根据数据描述，'Functional'缺失处认为是'Typ'
# # 'Utilities'在训练集中几乎全为'AllPub'，只有两个'NoSeWa'，而在测试集中有两处缺失，且在测试集中全为'AllPub'，可以认为该特征对预测没有帮助，删去
train['Functional'] = train['Functional'].fillna('Typ')
test['Functional'] = test['Functional'].fillna('Typ')

train.drop('Utilities', axis=1, inplace=True)
test.drop('Utilities', axis=1, inplace=True)

# 策略三：数字特征补零
zero_lists = ['GarageYrBlt', 'MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'GarageCars', 'GarageArea',
              'TotalBsmtSF']
for col in zero_lists:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)

# 策略四：数字特征非零补充中位数
train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
for ind in test['LotFrontage'][test['LotFrontage'].isnull().values==True].index:
    x = test['Neighborhood'].iloc[ind]
    test['LotFrontage'].iloc[ind] = train.groupby('Neighborhood')['LotFrontage'].median()[x]

# # 检查训练集是否存在缺失值
# print("训练集是否还存在缺失值:", train.isnull().sum().any())
# # 检查测试集是否存在缺失值
# print("测试集是否还存在缺失值:", test.isnull().sum().any())

# -----------------------------------------------------------------

# 7、转换类别特征
cate_features.remove('Utilities')
# print('The number of categorical features:', len(cate_features))

for col in cate_features:
    train[col] = train[col].astype(str)
    test[col] = test[col].astype(str)

# 策略一：对各类别存在顺序关系的类别特征采用LabelEncoder编码
le_features = ['Street', 'Alley', 'LotShape', 'LandContour', 'LandSlope', 'HouseStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual', 
               'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir',
               'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']
for col in le_features:
    encoder = LabelEncoder()
    value_train = set(train[col].unique())
    value_test = set(test[col].unique())
    value_list = list(value_train | value_test)
    encoder.fit(value_list)
    train[col] = encoder.transform(train[col])
    test[col] = encoder.transform(test[col])

# -----------------------------------------------------------------

# 8、处理偏斜特征
skewness = train[num_features].apply(lambda x: skew(x)).sort_values(ascending=False)
skewness = skewness[skewness>0.5]
skew_features = skewness.index
skewness

# # 输出偏斜特征及其偏斜度
# print("偏斜度较高的数值特征及其偏斜度：")
# print(skewness)

# 此处原作者代码出险问题，运行报错
# for col in skew_features:
#     lam = stats.boxcox_normmax(train[col]+1)    #+1是为了保证输入大于零
#     train[col] = boxcox1p(train[col], lam)
#     test[col] = boxcox1p(test[col], lam)

# -----------------------------------------------------------------
# 9、构建新的特征
train['IsRemod'] = 1 
train['IsRemod'].loc[train['YearBuilt']==train['YearRemodAdd']] = 0  #是否翻新(翻新：1， 未翻新：0)
train['BltRemodDiff'] = train['YearRemodAdd'] - train['YearBuilt']  #翻新与建造的时间差（年）

train['BsmtUnfRatio'] = 0
train['BsmtUnfRatio'].loc[train['TotalBsmtSF']!=0] = train['BsmtUnfSF'] / train['TotalBsmtSF']  #Basement未完成占总面积的比例

train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']  #总面积

#对测试集做同样的处理
test['IsRemod'] = 1 
test['IsRemod'].loc[test['YearBuilt']==test['YearRemodAdd']] = 0  #是否翻新(翻新：1， 未翻新：0)
test['BltRemodDiff'] = test['YearRemodAdd'] - test['YearBuilt']  #翻新与建造的时间差（年）
test['BsmtUnfRatio'] = 0
test['BsmtUnfRatio'].loc[test['TotalBsmtSF']!=0] = test['BsmtUnfSF'] / test['TotalBsmtSF']  #Basement未完成占总面积的比例
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']  #总面积

# -----------------------------------------------------------------
# 10、处理其余的类别特征
# 策略二：对不存在顺序关系的类别特征采用独热编码

# print('The shape of training data:', train.shape)
# print('The shape of testing data:', test.shape)

dummy_features = list(set(cate_features).difference(set(le_features)))
# print(dummy_features)

all_data = pd.concat((train.drop('SalePrice', axis=1), test)).reset_index(drop=True)
# print(all_data.shape)

all_data = pd.get_dummies(all_data, drop_first=True)  #注意独热编码生成的时候要去掉一个维度，保证剩下的变量都是相互独立的
# print(all_data.shape)

# -----------------------------------------------------------------
# 11、还原训练集与测试集并保存
trainset = all_data[:1458]
y = train['SalePrice']
trainset['SalePrice'] = y.values
testset = all_data[1458:]
print('The shape of training data:', trainset.shape)
print('The shape of testing data:', testset.shape)

trainset.to_csv('train_data.csv', index=False)
testset.to_csv('test_data.csv', index=False)


trainset.isnull().sum().any()

train_data = pd.read_csv('train_data.csv')
train_data.isnull().sum().any()