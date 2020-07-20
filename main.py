# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 16:10:12 2020

@author: Lyonel
"""

from rank import *
from cal import *

fund_nvalue_data_raw = pd.read_csv('./fundData/navAdj.csv')
fund = pd.read_csv('./fundData/fundInfo.csv', encoding='gbk', index_col=0)
rf_raw = pd.read_excel('./otherData/Rf.xlsx', header=0, sheet_name=0, index_col=0)
index_3 = pd.read_excel('./otherData/指数数据.xlsx', header=0, sheet_name=0, index_col=0)
dic_data = pd.read_csv('./otherData/dic_data.csv', index_col=0)
dic_data = dic_data.T.to_dict('list')
type_data = pd.read_excel('./otherData/评价指标列表.xlsx', header=1, sheet_name=0, index_col=1)
data1, data2 = dif_type(type_data)

'''
cl = '普通股票型基金'
ftype = '稳定性'
back_window = 36
'''
startTime = '2013-01-01'
endTime = '2020-06-30'

freq_r = 'm'
freq_sp = 'm'
type_0 = ['货币市场型基金']
type_1 = ['偏债混合型基金', '灵活配置型基金']

type_2 = data1.columns.to_list()
type_3 = data2.columns.to_list()
back_window2 = 0
fund_nvalue_data, rf_raw = change_index_form(fund_nvalue_data_raw, rf_raw)

for back_window in [12, 24, 36, 48, 60]:
    '''
    for cl in type_0:
        fil, ql = func_port(fund_nvalue_data, fund, index_3, rf_raw, cl, back_window, freq_r, freq_sp, startTime, endTime)
        for ftype in type_2:
            out, score_out = func_port2(fil, ql, data1, data2, cl, ftype)
            daily, dailyf, weekly, monthly, weeklyf, monthlyf, output = get_netvalue(fund_nvalue_data, out, dic_data, 200000000, 90)
            fof_index = func_score(dailyf, ret_cal(dailyf), rf_raw, ret_cal(index_3), datetime.strftime(dailyf.index[-1], "%Y-%m-%d"), back_window2, {'d': 250, 'w': 52, 'm': 12, 'y': 1}, freq_r, freq_sp)
            
            writer = pd.ExcelWriter('./货币市场型/'+str(back_window)+str(cl)+str(ftype)+'fof.xls')
            score_out.to_excel(writer,sheet_name="得分")
            output.to_excel(writer,sheet_name="权重及费率")
            dailyf.to_excel(writer,sheet_name="fof净值序列")
            fof_index.to_excel(writer,sheet_name="fof指标")
            writer.save()
    '''
    
    for cl in type_1:
        fil, ql = func_port(fund_nvalue_data, fund, index_3, rf_raw, cl, back_window, freq_r, freq_sp, startTime, endTime)
        for ftype in type_3:
            out, score_out = func_port2(fil, ql, data1, data2, cl, ftype)
            daily, dailyf, weekly, monthly, weeklyf, monthlyf, output = get_netvalue(fund_nvalue_data, out, dic_data, 200000000, 90)
            fof_index = func_score(dailyf, ret_cal(dailyf), rf_raw, ret_cal(index_3), datetime.strftime(dailyf.index[-1], "%Y-%m-%d"), back_window2, {'d': 250, 'w': 52, 'm': 12, 'y': 1}, freq_r, freq_sp)
            writer = pd.ExcelWriter('./股偏灵债型/'+str(back_window)+str(cl)+str(ftype)+'fof.xls')
            score_out.to_excel(writer,sheet_name="得分")
            output.to_excel(writer,sheet_name="权重及费率")
            dailyf.to_excel(writer,sheet_name="fof净值序列")
            fof_index.to_excel(writer,sheet_name="fof指标")
            writer.save()
            
