import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.model_selection import train_test_split
# 使用GridSearchCV进行参数搜索
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
# 使用GBDT
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import Imputer
# 绘制特征筛选
from xgboost import plot_importance
import matplotlib.pyplot as plt

dataset1 = pd.read_csv(
    r'C:\Users\XiaYang\Desktop\O2O\Data\ProcessDataSet1.csv')
# 标记标签量：没有消费返回0；收到优惠券15天内消费，返回1；收到优惠券超过15天消费返回-1
dataset1.label.replace(-1, 0, inplace=True)
dataset2 = pd.read_csv(
    r'C:\Users\XiaYang\Desktop\O2O\Data\ProcessDataSet2.csv')
dataset2.label.replace(-1, 0, inplace=True)
dataset3 = pd.read_csv(
    r'C:\Users\XiaYang\Desktop\O2O\Data\ProcessDataSet3.csv')

# 删除重复行数据
dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)
dataset12 = pd.concat([dataset1, dataset2], axis=0)
dataset12_y = dataset12.label
# 求正样本的比例
# print(dataset12_y.mean())
# dp.drop():删除行或列
# dataset12_x：52列
dataset12_x = dataset12.drop(
    ['user_id', 'label', 'day_gap_before', 'coupon_id', 'day_gap_after'], axis=1)
# dataset12_x = dataset12.drop(
#     ['user_id', 'label',  'coupon_id'], axis=1)
# print(dataset12_x.info())
# print(dataset12_x.head(5))

dataset3.drop_duplicates(inplace=True)
dataset3_preds = dataset3[['user_id', 'coupon_id', 'date_received']]
# dataset3_x：52列
dataset3_x = dataset3.drop(
    ['user_id', 'coupon_id', 'date_received', 'day_gap_before', 'day_gap_after'], axis=1)
# dataset3_x = dataset3.drop(
#     ['user_id', 'coupon_id', 'date_received'], axis=1)
# print(dataset3_x.info())
# print(dataset3_x.head(5))
# 打印所有特征
# features = list(dataset12_x.columns)
# print(list(zip(list(range(len(features))),features)))

# Xgboost训练的数据必须要使用xgb.DMatrix()转化后的形式
dataTrain = xgb.DMatrix(dataset12_x, label=dataset12_y)
dataTest = xgb.DMatrix(dataset3_x)

# 性能评价函数


def myauc(test):
    testgroup = test.groupby(['coupon_id'])
    aucs = []
    for i in testgroup:
        # i为tuple，i[0]为分组的标签，i[1]为每组中的元素
        tmpdf = i[1]
        # label必须有两类，如果只有一类，roc曲线中有一个数分母为0
        if len(tmpdf['label'].unique()) != 2:
            continue
        fpr, tpr, thresholds = roc_curve(
            tmpdf['label'], tmpdf['pred'], pos_label=1)
        aucs.append(auc(fpr, tpr))
    return np.average(aucs)

# ###############################################################################################
# 训练模型xgboost########################################################

params = {'booster': 'gbtree',
          'objective': 'binary:logistic',
          'eval_metric': 'auc',
          'gamma': 0.1,
          'min_child_weight': 1.1,
          'max_depth': 5,
          'lambda': 10,
          'subsample': 0.7,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 0.7,
          'eta': 0.01,
          'tree_method': 'exact',
          'seed':0
          }

# # params of wepon
# params = {'booster': 'gbtree',
#           'objective': 'binary:logistic',
#           'eval_metric': 'auc',
#           'gamma': 0,
#           'min_child_weight': 100,
#           'max_depth': 8,
#           'lambda': 50,
#           'subsample': 0.4,
#           'colsample_bytree': 0.6,
#           'eta': 0.008,
#           'tree_method': 'exact',
#           'base_score':0.11,
#           'seed':0
#           }

watchlist = [(dataTrain, 'train')]
# eval：获取返回值
model = xgb.train(params, dataTrain, num_boost_round=11689, evals=watchlist, early_stopping_rounds=50)

model.save_model(r'C:\Users\XiaYang\Desktop\O2O\Data\xgbmodel')

# 训练模型GBDT############################################################

# # params of wepon
# gbdtmodel = GradientBoostingClassifier(n_estimators=1000,
#                                        max_leaf_nodes = 32,
#                                        learning_rate = 0.1,
#                                        min_samples_leaf = 500,
#                                        random_state = 0,
#                                        subsample=0.6,
#                                        max_depth=8,
#                                        verbose=1,
#                                        n_iter_no_change = 50)

# params of 天池
gbdtmodel = GradientBoostingClassifier(n_estimators=1900,
                                       learning_rate = 0.01,
                                       min_samples_split = 200,
                                       min_samples_leaf = 50,
                                       random_state = 0,
                                       subsample=0.8,
                                       max_depth=9,
                                       verbose=1,
                                       n_iter_no_change = 50)
imp = Imputer(missing_values='NaN',strategy='mean',axis=0,verbose=0,copy=True)
imp_dataset12_x = imp.fit_transform(dataset12_x)
gbdtmodel.fit(imp_dataset12_x,dataset12_y)
s=pickle.dumps(gbdtmodel)
f=open(r'C:\Users\XiaYang\Desktop\O2O\Data\gbdtmodel','wb')
f.write(s)
f.close()

# ###############################################################################################
# 使用GridSearchCV进行参数搜索

param_test1 = {'max_depth': range(
    3, 10, 2), 'min_child_weight': range(1, 6, 2)}
# GridSearchCV（）：全部都是参数和模型，没有输入数据
# cv：交叉验证参数，n-fold数量
gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1,
                                               n_estimators=5,
                                               max_depth=5,
                                               min_child_weight=1,
                                               gamma=0,
                                               subsample=0.8,
                                               colsample_bytree=0.8,
                                               objective='binary:logistic',
                                               scale_pos_weight=1,
                                               seed=0),
                       param_grid=param_test1,
                       scoring='roc_auc',
                       iid=False,
                       cv=3)
# gsearch.fit：根据模型和输入数据，拟合参数
gresult = gsearch.fit(dataset12_x, dataset12_y)
# 显示参数
gmeans = gresult.cv_results_['mean_test_score']
gparams = gresult.cv_results_['params']
# zip：将两个列表对应位置拟合成一个列表，返回列表
for param, mean in zip(gparams, gmeans):
    print("%s  with:   %s" % (param, mean))
print('The best params is: %s, the auc is %s' %
      (gresult.best_params_, gresult.best_score_))
# 运行结果
# {'max_depth': 3, 'min_child_weight': 1}  with:   0.8057366828744567
# {'max_depth': 8, 'min_child_weight': 1}  with:   0.8596112938323478
# The best params is: {'max_depth': 8, 'min_child_weight': 1}, the auc is 0.8596112938323478

# ###############################################################################################
# 测试集预测

# xgboost
# xgb.Booster:定义一个xgb模型
model = xgb.Booster()
model.load_model(r'C:\Users\XiaYang\Desktop\O2O\Data\xgbmodel11689_binary')
# predict test set
dataset3_preds1 = dataset3_preds
dataset3_preds1['label'] = model.predict(dataTest)
# 标签归一化在[0，1]
dataset3_preds1.label = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(
    dataset3_preds1.label.values.reshape(-1, 1))

dataset3_preds1.sort_values(by=['coupon_id', 'label'], inplace=True)
dataset3_preds1.to_csv(
    r"C:\Users\XiaYang\Desktop\O2O\Data\xgb_preds.csv", index=None, header=None)
print(dataset3_preds1.describe())

# gbdtmodel1900_tianchi_early
f2=open(r'C:\Users\XiaYang\Desktop\O2O\Data\gbdtmodel1900_tianchi_early','rb')
s2=f2.read()
gbdtmodel=pickle.loads(s2)
f2.close()
imp = Imputer(missing_values='NaN',strategy='mean',axis=0,verbose=0,copy=True)
imp_dataTest = imp.fit_transform(dataset3_x)
# predict test set
dataset3_preds1 = dataset3_preds
dataset3_preds1['label'] = gbdtmodel.predict_proba(imp_dataTest)[:,1]

dataset3_preds1.sort_values(by=['coupon_id', 'label'], inplace=True)
dataset3_preds1.to_csv(
    r"C:\Users\XiaYang\Desktop\O2O\Data\gbdt_preds.csv", index=None, header=None)
print(dataset3_preds1.describe())

# ###############################################################################################
# 训练集auc值计算

# # XGBOOST
# model = xgb.Booster()
# model.load_model(r'C:\Users\XiaYang\Desktop\O2O\Data\xgbmodel11689_binary')
#
# temp = dataset12[['coupon_id', 'label']].copy()
# temp['pred'] = model.predict(xgb.DMatrix(dataset12_x))
# temp.pred = MinMaxScaler(copy=True, feature_range=(
#     0, 1)).fit_transform(temp['pred'].values.reshape(-1, 1))
# print(myauc(temp))
#
# # # GBDT
# # f2=open(r'C:\Users\XiaYang\Desktop\O2O\Data\gbdtmodel1900_tianchi_early','rb')
# # s2=f2.read()
# # gbdtmodel=pickle.loads(s2)
# # f2.close()
# # imp = Imputer(missing_values='NaN',strategy='mean',axis=0,verbose=0,copy=True)
# # imp_dataset12_x = imp.fit_transform(dataset12_x)
# # temp = dataset12[['coupon_id', 'label']].copy()
# # temp['pred'] = gbdtmodel.predict_proba(imp_dataset12_x)[:,1]
# # print(myauc(temp))

# ###############################################################################################
# ################################模型融合#############################################

# 训练集auc值计算（模型融合）


temp = dataset12[['coupon_id', 'label']].copy()
preds = pd.DataFrame()

# # xgbmodel3500_norm_binary
# model = xgb.Booster()
# model.load_model(r'C:\Users\XiaYang\Desktop\O2O\Data\xgbmodel3500_norm_binary')
# preds['pred1'] = model.predict(xgb.DMatrix(dataset12_x))
# preds.pred1 = MinMaxScaler(copy=True, feature_range=(
#     0, 1)).fit_transform(preds['pred1'].values.reshape(-1, 1))
# print('---------------xgbmodel3500_norm_binary predict done-------------------')

# xgbmodel11689_binary
model = xgb.Booster()
model.load_model(r'C:\Users\XiaYang\Desktop\O2O\Data\xgbmodel11689_binary')
preds['pred2'] = model.predict(xgb.DMatrix(dataset12_x))
preds.pred2 = MinMaxScaler(copy=True, feature_range=(
    0, 1)).fit_transform(preds['pred2'].values.reshape(-1, 1))
print('---------------xgbmodel11689_binary predict done-------------------')

# xgbmodel3500_rank
model = xgb.Booster()
model.load_model(r'C:\Users\XiaYang\Desktop\O2O\Data\xgbmodel3500_rank')
preds['pred3'] = model.predict(xgb.DMatrix(dataset12_x))
preds.pred3 = MinMaxScaler(copy=True, feature_range=(
    0, 1)).fit_transform(preds['pred3'].values.reshape(-1, 1))
print('---------------xgbmodel3500_rank predict done-------------------')

# xgbmodel6558_binary
model = xgb.Booster()
model.load_model(r'C:\Users\XiaYang\Desktop\O2O\Data\xgbmodel6558_binary')
preds['pred4'] = model.predict(xgb.DMatrix(dataset12_x))
preds.pred4 = MinMaxScaler(copy=True, feature_range=(
    0, 1)).fit_transform(preds['pred4'].values.reshape(-1, 1))
print('---------------xgbmodel6558_binary predict done-------------------')

# gbdtmodel1900_tianchi_early
f2 = open(r'C:\Users\XiaYang\Desktop\O2O\Data\gbdtmodel1900_tianchi_early', 'rb')
s2 = f2.read()
gbdtmodel = pickle.loads(s2)
f2.close()
imp = Imputer(missing_values='NaN', strategy='mean',
              axis=0, verbose=0, copy=True)
imp_dataset12_x = imp.fit_transform(dataset12_x)
preds['pred5'] = gbdtmodel.predict_proba(imp_dataset12_x)[:, 1]
print('---------------gbdtmodel1900_tianchi_early predict done-------------------')

# gbdtmodel1000_wepon_early
f3 = open(r'C:\Users\XiaYang\Desktop\O2O\Data\gbdtmodel1000_wepon_early', 'rb')
s3 = f3.read()
gbdtmodel = pickle.loads(s3)
f3.close()
imp = Imputer(missing_values='NaN', strategy='mean',
              axis=0, verbose=0, copy=True)
imp_dataset12_x = imp.fit_transform(dataset12_x)
preds['pred6'] = gbdtmodel.predict_proba(imp_dataset12_x)[:, 1]
print('---------------gbdtmodel1000_wepon_early predict done-------------------')

# 平均
# temp['pred'] = preds.mean(1).values.copy()
# 加权平均
temp['pred'] = preds['pred2'].values.copy()*0.1 + preds['pred3'].values.copy()*0.25 + preds['pred4'].values.copy() * \
    0.45 + preds['pred5'].values.copy()*0.1 + preds['pred6'].values.copy()*0.1
print(myauc(temp))

# ###############################################################################################
# 测试集预测（模型融合）

# dataset3_preds只有'user_id', 'coupon_id', 'date_received'三列
dataset3_preds1 = dataset3_preds
testpreds = pd.DataFrame()

# xgbmodel3500_rank
model = xgb.Booster()
model.load_model(r'C:\Users\XiaYang\Desktop\O2O\Data\xgbmodel3500_rank')
testpreds['label3'] = model.predict(dataTest)
testpreds.label3 = MinMaxScaler(copy=True, feature_range=(
    0, 1)).fit_transform(testpreds['label3'].values.reshape(-1, 1))
print('---------------xgbmodel3500_rank predict done-------------------')

# xgbmodel11689_binary
model = xgb.Booster()
model.load_model(r'C:\Users\XiaYang\Desktop\O2O\Data\xgbmodel11689_binary')
testpreds['label5'] = model.predict(dataTest)
testpreds.label5 = MinMaxScaler(copy=True, feature_range=(
    0, 1)).fit_transform(testpreds['label5'].values.reshape(-1, 1))
print('---------------xgbmodel11689_binary predict done-------------------')

# xgbmodel6558_binary
model = xgb.Booster()
model.load_model(r'C:\Users\XiaYang\Desktop\O2O\Data\xgbmodel6558_binary')
testpreds['label1'] = model.predict(dataTest)
testpreds.label1 = MinMaxScaler(copy=True, feature_range=(
    0, 1)).fit_transform(testpreds['label1'].values.reshape(-1, 1))
print('---------------xgbmodel6558_binary predict done-------------------')

# gbdtmodel1900_tianchi_early
f2 = open(r'C:\Users\XiaYang\Desktop\O2O\Data\gbdtmodel1900_tianchi_early', 'rb')
s2 = f2.read()
gbdtmodel = pickle.loads(s2)
f2.close()
imp = Imputer(missing_values='NaN', strategy='mean',
              axis=0, verbose=0, copy=True)
imp_dataTest = imp.fit_transform(dataset3_x)
testpreds['label2'] = gbdtmodel.predict_proba(imp_dataTest)[:, 1]
print('---------------gbdtmodel1900_tianchi_early predict done-------------------')

# gbdtmodel1000_wepon_early
f2 = open(r'C:\Users\XiaYang\Desktop\O2O\Data\gbdtmodel1000_wepon_early', 'rb')
s2 = f2.read()
gbdtmodel = pickle.loads(s2)
f2.close()
imp = Imputer(missing_values='NaN', strategy='mean',
              axis=0, verbose=0, copy=True)
imp_dataTest = imp.fit_transform(dataset3_x)
testpreds['label4'] = gbdtmodel.predict_proba(imp_dataTest)[:, 1]
print('---------------gbdtmodel1000_wepon_early predict done-------------------')


dataset3_preds1['label'] = testpreds['label1'].values.copy()*0.45 + testpreds['label2'].values.copy() * \
    0.1 + testpreds['label3'].values.copy()*0.25 + testpreds['label4'].values.copy() * \
    0.1 + testpreds['label5'].values.copy()*0.1
dataset3_preds1.sort_values(by=['coupon_id', 'label'], inplace=True)
dataset3_preds1.to_csv(
    r"C:\Users\XiaYang\Desktop\O2O\Data\xgb_preds.csv", index=None, header=None)
print(dataset3_preds1.describe())

# ################################模型融合#############################################
# ###############################################################################################

# 最大迭代次数调优：使用xgb.cv

params={'booster':'gbtree',
	    'objective': 'binary:logistic',
	    'eval_metric':'auc',
	    'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.01,
	    'tree_method':'exact'
	    }
# nfold：每次取1/10作为验证集，10个验证集的auc取平均作为验证集的auc
# cvresult：num_boost_round * 4 的dataframe
# 四列为train-aue-mean, train-auc-std, test-auc-mean, test-auc-std
cvresult = xgb.cv(params, dataTrain, num_boost_round=20000, nfold=4, metrics='auc', seed=0, callbacks=[
            xgb.callback.print_evaluation(show_stdv=False),
            xgb.callback.early_stop(50)
        ])
# cvresult的行数为从头的最优迭代次数；可能训练到了11739次，但是返回的是最优的11689行（中间正好差50）
num_round_best = cvresult.shape[0] - 1
print('Best round num: ', num_round_best)

watchlist = [(dataTrain,'train')]
model1 = xgb.train(params,dataTrain,num_boost_round=num_round_best,evals=watchlist)

model1.save_model(r'C:\Users\XiaYang\Desktop\O2O\Data\xgbmodel1')
print('------------------------train done------------------------------')

###############################################################################################
# Xgboost特征筛选功能

model1 = xgb.Booster()
model1.load_model(r'C:\Users\XiaYang\Desktop\O2O\Data\xgbmodel6558_binary')
# model1.get_fscore()：
# weight - 该特征在所有树中被用作分割样本的特征的次数。
# gain - 在所有树中的平均增益。
# cover - 在树中使用该特征时的平均覆盖范围。(还不是特别明白)
feature_score = model1.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=False)  # value逆序排序

# 将特征写入文件
# fs = []
# for (key, value) in feature_score:
#     fs.append("{0},{1}\n".format(key, value))
#
# with open('D:/MachineLearning/ofo/ofoOptimization/xgb_feature_score.csv', 'w') as f:
#     f.writelines("feature,score\n")
#     f.writelines(fs)

df = pd.DataFrame(feature_score , columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

plt.figure()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.show()

# # plot_importance:xgboost内置绘制特征函数
# plt.figure()
# plot_importance(model1)
# plt.show()

# ###############################################################################################
