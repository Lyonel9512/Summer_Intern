# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 09:49:18 2020

@author: Lyonel
"""

import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from statsmodels import regression
import statsmodels.api as sm

def to_date(x):  # 修改日期小函数
    x = list(str(x))
    x.insert(4, '-')
    x.insert(7, '-')
    x = "".join(x)
    return x

def change_index_form(x, y):
    data = x.copy()
    data.index = data.iloc[:, 0].apply(lambda x: to_date(x))  # 对净值表里的日期数据进行格式调整，以便于调为时间戳格式
    data.index = pd.to_datetime(data.index)
    data.drop(columns=data.columns[0], inplace=True)  # 将index设置为日期的时间戳格式
    data.fillna(method='ffill', inplace=True)
    
    data2 = y.copy()

    data2 = pd.Series(data2.iloc[:, 0].values, index=data2.index, copy=True)  # 无风险利率列表
    return data, data2

def mdd(returns):  # 最大回撤计算
    returns = list(returns)
    max_draw_down = 0
    temp_max_value = 0
    for i in range(1, len(returns)):
        temp_max_value = max(temp_max_value, returns[i-1])
        max_draw_down = min(max_draw_down, returns[i]/temp_max_value - 1)
    return max_draw_down

def ret_cal(values):
    return values/values.shift(1) - 1

def clean_data(nv_df, fund_detail, ind, clist):
    # 清洗基金基本信息
    nv_df_c = nv_df.copy()
    fund_detail = fund_detail.copy()
    fund_detail.ivstTypeInDt = fund_detail.ivstTypeInDt.apply(
        lambda x: to_date(x)[:-2])
    fund_detail.ivstTypeExDt = fund_detail.ivstTypeExDt.apply(
        lambda x: to_date(x)[:-2])
    fund_detail.ivstTypeInDt = pd.to_datetime(fund_detail.ivstTypeInDt)
    fund_detail.ivstTypeExDt = pd.to_datetime(fund_detail.ivstTypeExDt)
    fund_detail_filt = fund_detail[fund_detail['scndIvstType'] == clist]
    fdf = fund_detail_filt.copy()
    # 'ivstTypeInDt'和 'ivstTypeExDt' 控制了基金处于该时间范围内是某种分类
    # 将NaT格式转为当前时点表示无穷点，筛选出基金进入该分类的时间早于t_0且退出该分类的时间晚于t的基金
    fdf.loc[:, 'ivstTypeInDt'] = fdf['ivstTypeInDt'].replace(pd.NaT, datetime.now())
    fdf.loc[:, 'ivstTypeExDt'] = fdf['ivstTypeExDt'].replace(pd.NaT, datetime.now())
    
    # 调整指数序列为时间序列
    ind.index = pd.to_datetime(ind.index)
    
    r_ind_all = ret_cal(ind)

    r_w_all = ret_cal(nv_df_c)
    anufreq_dic = {'d': 250, 'w': 52, 'm': 12, 'y': 1}  # 频率换算字典，用于年化
    
    return fdf, r_ind_all, nv_df_c, r_w_all, anufreq_dic
# 变量分别为：基金净值序列，无风险利率序列，指数序列，基金详情序列，时间点，回溯窗口，选择的基金种类列表标准差参数，夏普比参数


def nv_w_set(nv_df_c, fdf, t, back_window):
    #startTime = time.clock()
    # 转变时间类型
    # t_0是回溯期开始往前推6个月
    t_0 = datetime.strptime(t, "%Y-%m-%d") - \
        relativedelta(months=back_window + 6)
    # 先筛选出在列表内的基金
    
    # fund_detail_filt2 是最终符合标准的基金列表
    fund_detail_filt2 = fdf[(fdf['ivstTypeInDt'] < t_0)
                            & (fdf['ivstTypeExDt'] > datetime.strptime(t, "%Y-%m-%d"))].index.to_list()
    # 选取基金净值序列在时间窗口内且为上面筛选出的基金
    nv_df_w = nv_df_c.loc[:, fund_detail_filt2]
    return nv_df_w

def func_score(nv_df_w, r_w_all, rf, r_ind_all, t, back_window, anufreq_dic, freq_r, freq_sp):
    # 最终输出df格式
    nv_df_w = pd.DataFrame(nv_df_w) if type(nv_df_w) == pd.Series else nv_df_w
    ini_t = datetime.strptime(t, "%Y-%m-%d") - relativedelta(months=back_window)
    nv_df_w = nv_df_w.loc[ini_t.strftime("%Y-%m-%d"):t, :] if back_window != 0 else nv_df_w.loc[:t, :]  # 若回溯月份为0则是从头开始
    r_w = pd.DataFrame(r_w_all).loc[nv_df_w.index, nv_df_w.columns] if type(r_w_all) == pd.Series else r_w_all.loc[nv_df_w.index, nv_df_w.columns]  # 日收益率列表
    out = pd.DataFrame(index=['年化收益率', '收益率标准差', '最大回撤', 'Sharpe比率', 'Calmar比率', 'Alpha', 'Alpha稳定性'], columns=nv_df_w.columns)
    # 指标的收益率计算
    r_ind = r_ind_all.loc[nv_df_w.index, :]
    # 最终结束日期，因为要用于计算真实日期
    end_t = datetime.strptime(t, "%Y-%m-%d")
    # 计算真实经过日期
    delta_t = nv_df_w.apply(lambda x: (
        end_t - x.dropna().index[0]).days + 1 if len(x.dropna()) > 0 else 0, axis=0)
    nv_r_w = ret_cal(nv_df_w.resample(freq_r).last()) if freq_r != 'd' else r_w  # 用于标准差计算的nv
    nv_rs_w = ret_cal(nv_df_w.resample(freq_sp).last()) if freq_sp != 'd' else r_w  # 用于sharpe计算的nv
    std = nv_r_w.apply(lambda x: x.dropna().std() *
                       ((anufreq_dic[freq_r]) ** (1/2)), axis=0)  # 转化为年化
    spr_std = nv_rs_w.apply(lambda x: x.dropna().std()
                            * ((anufreq_dic[freq_sp]) ** (1/2)), axis=0)  # 转化为年化
    rfm = nv_df_w.apply(lambda x: rf[x.dropna().index]).mean()/100  # 对应时间内无风险利率均值
    m = r_w.apply(lambda x: mdd((x.fillna(0)+1).cumprod()))  # 最大回撤
    # 在每一列下面加入真实经过日期用于计算
    nv_df_w1 = nv_df_w.append(delta_t, ignore_index=True)
    # 年化收益率采用真实日和365一年计算
    anu_r = nv_df_w1.apply(lambda x: (
        x.dropna().iloc[-2]/x.dropna().iloc[0]) ** (365/x.dropna().iloc[-1]) - 1 if len(x.dropna()) > 1 else 0, axis=0)  # 计算年化收益率
    c = anu_r/m  # calmar比率
    spr = (anu_r - rfm)/spr_std  # sharpe比率
    # 先对每个收益率和指数做回归，存储进model,这样model里就是每个基金的回归结果
    model = r_w.apply(lambda x: regression.linear_model.OLS(x.dropna(), sm.add_constant(
        r_ind.loc[x.dropna().index])).fit() if len(x.dropna()) > 0 else float('nan'), axis=0)
    # 调取每个model里的截距项
    alpha = model.fillna(0).apply(
        lambda x: x.params[0] if x != 0 else float('nan'))
    # 调取每个model里的残差项并计算标准差
    res_std = model.fillna(0).apply(lambda x: x.resid.std()
                                    if x != 0 else float('nan'))

    out.loc['年化收益率'] = anu_r
    out.loc['收益率标准差'] = std
    out.loc['最大回撤'] = m
    out.loc['Sharpe比率'] = spr
    out.loc['Calmar比率'] = c * -1
    out.loc['Alpha'] = alpha
    out.loc['Alpha稳定性'] = res_std
    #endTime = time.clock()
    #print(endTime - startTime)
    return out


def dif_type(data):
    data.drop(columns = data.columns[0], inplace =  True)
    data.index = ['年化收益率', '收益率标准差', '最大回撤', 'Sharpe比率', 'Calmar比率', 'Alpha', 'Alpha稳定性']
    data1 = data.iloc[:, :4]
    data2 = data.iloc[:, 4:]
    return data1, data2

def func_rank(input_data, cl, ftype, data1, data2):
    # 设定排名，最大回撤是负值，所以是倒序排列，calmar乘以-1所以正序排列。
    rank_data = input_data.rank(axis=1)
    # 排名标准化作为得分
    score_data = rank_data.apply(lambda x: x * 100 /len(x.dropna()), axis = 1)
    #分数 = Σ排名*权重
    if cl == '货币市场型基金':
        score  = score_data.apply(lambda x: (x * data1.loc[:, ftype]).sum()/100, axis = 0)
    elif cl in ['普通股票型基金', '被动指数型基金', '偏股混合型基金', '增强指数型基金', '灵活配置型基金', '偏债混合型基金']:
        score  = score_data.apply(lambda x: (x * data2.loc[:, ftype]).sum()/100, axis = 0)
    # 分数越小排名越高
    rank = score.sort_values(ascending = False)
    out = score_data.T.reset_index()
    return rank.drop_duplicates().iloc[:10].index.to_list(), out

#输出权重表
# 变量分别为：基金净值序列，无风险利率序列，指数序列，基金详情序列，回溯窗口，选择的基金种类列表标准差参数，夏普比参数，起始日，终结日
def func_port(nv_df, fund_detail, ind, rf_raw, clist, back_window, freq_r, freq_sp, s_d, e_d):
    # 设一个从s_d到e_d之间的季度最后一天数值，该天数序列从nv_df的索引获得，这样得到的每季度末是交易日
    #清洗数据放在此处，避免多重循环调用
    fdf, r_ind_all, nv_df_c, r_w_all, anufreq_dic = clean_data(nv_df, fund_detail, ind, clist)
    longdate = pd.Series(nv_df_c.index)
    longdate.index = longdate
    #如果给定的e_d太迟，则用nv_df最后一天作为e_d，防止给出超过时间序列的配仓
    e_d = nv_df_c.index[-1] if datetime.strptime(e_d, "%Y-%m-%d") > nv_df_c.index[-1] else e_d
    q_last = longdate.resample('q').last().loc[s_d:e_d].to_list()
    # 每季末回溯过去三年得到一个前十名的排名
    fund_ind_l = []
    for q in q_last:
        fund_ind = func_score(nv_w_set(nv_df_c, fdf, datetime.strftime(q, "%Y-%m-%d"), back_window), r_w_all, rf_raw, r_ind_all, datetime.strftime(q, "%Y-%m-%d"), back_window, anufreq_dic, freq_r, freq_sp)
        fund_ind_l.append(fund_ind)
    return fund_ind_l, q_last

def func_port2(fund_ind_l, q_last, data1, data2, clist, ftype):
    out = pd.DataFrame(columns=['日期', '代码', '权重'])
    score_df_l = []
    for i in range(len(fund_ind_l)):
        code_list, score_df = func_rank(fund_ind_l[i], clist, ftype, data1, data2)
        l_c = len(code_list)
        for l in range(l_c):
            out = out.append(
                {'日期': q_last[i], '代码': code_list[l], '权重': 1/l_c}, ignore_index=True)
        score_df.index = [q_last[i]] * len(score_df)
        score_df_l.append(score_df)
    out.set_index('日期', inplace = True, drop = True)
    score_out = pd.concat(score_df_l)
    return out, score_out
