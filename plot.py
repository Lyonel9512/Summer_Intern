# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:49:34 2020

@author: Lyonel
"""

import seaborn as sns
from rank import *
import matplotlib.pyplot as plt

def plot_df(data_1, data_2, cl, ftype, back_window, method):
    data_1 = data_1.astype('float')
    data_2 = data_2.astype('float')
    plt.figure(figsize=(15, 10))
    plt.title(u"各优选策略净值对比")
    plt.ylabel(u"净值")
    sns.lineplot(data = data_1)
    plt.xlabel(u"时间")
    if method == 'cl':
        plt.savefig('./输出图表/策略对比/' +str(cl)+str(back_window/12) + '年.png')
        data_2.to_excel('./输出图表/策略对比/'+str(back_window) + str(cl) +'fof_index.xls')
    elif method == 'ftype':
        plt.savefig('./输出图表/基金对比/' +str(ftype)+str(back_window/12) + '年.png')
        data_2.to_excel('./输出图表/基金对比/'+str(back_window) + str(ftype) +'fof_index.xls')
    
def output(type_0_1, data1_2, back_window, intype, bm):
    comparedf = []
    comparedf2 = []
    for cl in type_0_1:
        for ftype in data1_2.columns:
            if intype == '货币市场型':
                os = './货币市场型/' + str(back_window) + str(cl) + str(ftype) + 'fof.xls'
            elif intype == '股偏灵债型':
                os = './股偏灵债型/' + str(back_window) + str(cl) + str(ftype) + 'fof.xls'
            data_nv = pd.read_excel(os, header=0, sheet_name=2, index_col=0)
            data_nv = data_nv.append([cl])
            data_nv = data_nv.append([ftype])
            data_index = pd.read_excel(os, header=0, sheet_name=3, index_col=0)
            data_index = data_index.append([cl])
            data_index = data_index.append([ftype])
            data_nv = pd.Series(data_nv.iloc[:,0])
            data_index = pd.Series(data_index.iloc[:,0])
            data_nv.name = str(int(back_window/12)) + '年回溯期' + str(cl) + str(ftype) + '优选策略净值序列'
            data_index.name = str(int(back_window/12)) + '年回溯期' + str(cl) + str(ftype) + '优选策略净值指标'
            
            comparedf.append(data_nv)
            comparedf2.append(data_index)
    comparedf = pd.concat(comparedf, axis = 1)
    comparedf2 = pd.concat(comparedf2, axis = 1)
    
    
    
    for cl in type_0_1:
        df = comparedf.loc[:,comparedf.iloc[-2,:] == cl].iloc[:-2,:]
        df[cl+'基准'] = bm.loc[:,cl+'基准']
        df2 = comparedf2.loc[:,comparedf2.iloc[-2,:] == cl].iloc[:-2,:]
        plot_df(df, df2, cl, ftype, back_window, 'cl')
    for ftype in data1_2.columns:
        df = comparedf.loc[:,comparedf.iloc[-1,:] == ftype].iloc[:-2,:]
        df2 = comparedf2.loc[:,comparedf2.iloc[-1,:] == ftype].iloc[:-2,:]
        plot_df(df, df2, cl, ftype, back_window, 'ftype')

    
sns.set(font='SimHei')
type_data = pd.read_excel('./otherData/评价指标列表.xlsx', header=1, sheet_name=0, index_col=1)
data1, data2 = dif_type(type_data)

type_0 = ['货币市场型基金']
type_1 = ['灵活配置型基金', '偏债混合型基金']

type_2 = data1.columns.to_list()
type_3 = data2.columns.to_list()



benchmark_ls = []
for i in type_1:
    bm = pd.read_excel('./otherData/'+i+'.xls', header = 0, index_col = 0)
    bm = pd.Series(bm.iloc[:,0]/bm.iloc[0,0])
    bm.name = i + '基准'
    benchmark_ls.append(bm)
bm = pd.read_excel('./otherData/'+'货币市场型基金'+'.xls', header = 0, index_col = 0)
bm = bm['2013-03-29':'2020-06-30']
bm = pd.Series(bm.iloc[:,0]/bm.iloc[0,0])
bm.name = '货币市场型基金' + '基准'

benchmark_ls.append(bm)
bm = pd.concat(benchmark_ls, axis = 1)


for back_window in [12, 24, 36, 48, 60]:
    output(type_0, data1, back_window, '货币市场型', bm)
    output(type_1, data2, back_window, '股偏灵债型', bm)
