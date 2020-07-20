# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 19:45:46 2020

@author: Lyonel
"""
import re
import pandas as pd
#import matplotlib.pyplot as plt


def func(x):  # 用于判断所处区间
    for i, a in enumerate(x):
        if a < 0:
            return i


def calculate(dic_data, fund_id, buy, hold):  # 输入字典，基金id，购买额，持有期限
    offer_list = dic_data[fund_id][0]  # 获得申购信息
    redeem_list = dic_data[fund_id][1]  # 获得赎回信息
    offer_list = [offer_list.split('],')[0], offer_list.split('],')[
        1]]  # 只需要第一顺位的投资者类别
    redeem_list = [redeem_list.split('],')[0], redeem_list.split('],')[1]]
    o_sec = [float(s) for s in re.findall(r"\d+\.?\d*|inf",
                                          offer_list[0])]  # 由于字典里存储的是字符串，转化为列表形式
    o_fee = [float(s) for s in re.findall(r"\d+\.?\d*", offer_list[1])]
    r_sec = [float(s) for s in re.findall(r"\d+\.?\d*|inf", redeem_list[0])]
    r_fee = [float(s) for s in re.findall(r"\d+\.?\d*", redeem_list[1])]
    f1 = o_fee[func([buy - s for s in o_sec])]  # 代入func函数输出相应费率
    f2 = r_fee[func([hold - s for s in r_sec])]
    return f1, f2


def get_netvalue(fund_data, port_inf, dic, b, h):  # 输入data为净值表，port_inf为月度调仓信息
    data = fund_data.copy()
    p = port_inf.copy()
    p['申购费率'] = ''
    p['赎回权重'] = ''
    p['赎回费率'] = ''
    p = p.set_index([p.index, '代码'])
    port_date = port_inf.index.drop_duplicates().to_list()  # 调仓日期
    port_df = pd.DataFrame(index=data.index, columns=port_inf['代码'].drop_duplicates(
    ).to_list())  # 每日持仓比的dataframe
    for date in port_date:
        for fund in port_inf.loc[date, '代码']:
            port_df.loc[date, fund] = port_inf.loc[date].loc[port_inf.loc[date,
                                                                          '代码'] == fund]['权重'].values[0]
        # 将调仓日每个基金的持仓比重填入日持仓比表格，并将同一天没有持仓的基金比重设置为0
        port_df.loc[date, :].fillna(0, inplace=True)

    port_df.fillna(method='ffill', inplace=True)  # 向下填空缺值，这样调仓后仓位保持一个月

    port_df.dropna(inplace=True)  # 去除调仓以前的天数信息

    # 从总净值表中获取调仓以来日期和调仓包括的基金每日净值数据
    data_value = data.loc[port_df.index, port_df.columns]
    rlist = []  # 空列表存储每个仓位期间收益率
    rflist = [] # rf储存包括费率的净值表
    # of和re是查找给定参数的申购赎回费率列表
    of_ss = pd.Series(index=port_df.columns, data=[(calculate(dic, f, b, h)[
                      0])/(b * 10000) if calculate(dic, f, b, h)[0] > 1 else calculate(dic, f, b, h)[0] for f in port_df.columns])
    red_ss = pd.Series(index=port_df.columns, data=[(calculate(dic, f, b, h)[
                       1])/(b * 10000) if calculate(dic, f, b, h)[1] > 1 else calculate(dic, f, b, h)[1] for f in port_df.columns])
    for i in range(len(port_date)):  # 选取调仓日
        #the_first = True if i == 0 else False
        #the_last = False
        if i == len(port_date) - 1:  # 最后一个周期和前面取值方法不同
            #the_last = True
            w = data_value.loc[port_date[i]:].dropna(how='all', axis=1)
            d = pd.DataFrame(port_df.loc[port_date[i]:, w.columns], copy=True)
        else:
            w = data_value.loc[port_date[i]:port_date[i+1]
                               ].dropna(how='all', axis=1)  # 每个调仓周期净值表
            d = pd.DataFrame(
                port_df.loc[port_date[i]:port_date[i+1], w.columns], copy=True)  # 调仓周期权重表
        w = w.apply(lambda x: x.dropna()/x.dropna()[0], axis=0)  # 该周期内净值归一化
        w1 = w.copy()
        w1.iloc[0, :] = w1.iloc[0, :] * (1 + of_ss)  # 月底申购费率对净值影响
        l = p.loc[d.index[0], ].index.to_list()
        d.iloc[-1] = d.iloc[-2] if len(d) > 1 else d.iloc[-1]# 每月最后一天仍然保有相同权重
        wd = w * d  # 组合内购买基金当月净值变化
        v = (wd).sum(axis=1)  # 无费率影响权重乘以归一后净值加总得到当月组合净值
        mon_w = wd.iloc[-1]/v[-1]  # 无费率影响月底组合内净值占比
        w1.iloc[-1, :] = w1.iloc[-1, :] * (1 - red_ss)  # 月底赎回费率对净值影响计算
        w1d = w1 * d  # 有费率影响组合净值变化（包含月初月末的
        # 有费率影响计算组合当月收益率
        rf = ((w1d).sum(axis=1)/(w1d).sum(axis=1).shift(1) - 1)[1:]
        # 无费率影响计算组合收益率
        r = ((wd).sum(axis=1)/(wd).sum(axis=1).shift(1) - 1)[1:]
        rflist.append(rf)
        rlist.append(r)  # 将收益率存储进列表
        for fid in l:
            p_temp = p.loc[d.index[0], fid].copy()
            p_temp.loc['申购费率'] = of_ss.loc[fid]
            p_temp.loc['赎回权重'] = mon_w.loc[fid]
            p_temp.loc['赎回费率'] = red_ss.loc[fid]
            p.loc[d.index[0], fid] = p_temp

    return_daily = pd.concat(rlist)  # 拼接收益率
    returnf_daily = pd.concat(rflist)
    returnf_daily[port_df.index[0]] = 0
    return_daily[port_df.index[0]] = 0
    return_daily.sort_index(inplace=True)  # 第一天收益率修正为0
    returnf_daily.sort_index(inplace=True)
    nvalue_daily = (return_daily + 1).cumprod()  # 组合收益率+1累乘为组合净值
    nvaluef_daily = (returnf_daily + 1).cumprod()

    nvalue_weekly = nvalue_daily.resample(
        'w').last().dropna()  # 重采样每周最后一天制成周度净值

    nvalue_monthly = nvalue_daily.resample(
        'm').last().dropna()  # 重采样每月最后一天制成月度净值

    nvaluef_weekly = nvaluef_daily.resample(
        'w').last().dropna()  # 重采样每周最后一天制成周度净值

    nvaluef_monthly = nvaluef_daily.resample(
        'm').last().dropna()  # 重采样每月最后一天制成月度净值

    p = p.reset_index('代码')
    p.iloc[:, 2:] = p.iloc[:, 2:].astype('float')
    p.rename(columns={'权重': '申购权重'}, inplace=True)
    return nvalue_daily, nvaluef_daily, nvalue_weekly, nvalue_monthly, nvaluef_weekly, nvaluef_monthly, p  # 输出结果
