# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 19:45:46 2020

@author: Lyonel
"""
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from statsmodels import regression
import statsmodels.api as sm
import re
#import matplotlib.pyplot as plt


def to_date(x):  # 修改日期小函数
    x = list(str(x))
    x.insert(4, '-')
    x.insert(7, '-')
    x = "".join(x)
    return x


def mdd(returns):  # 最大回撤计算
    returns = list(returns)
    max_draw_down = 0
    temp_max_value = 0
    for i in range(1, len(returns)):
        temp_max_value = max(temp_max_value, returns[i-1])
        max_draw_down = min(max_draw_down, returns[i]/temp_max_value - 1)
    return max_draw_down

# 变量分别为：基金净值序列，无风险利率序列，指数序列，基金详情序列，时间点，回溯窗口，选择的基金种类列表标准差参数，夏普比参数

def func(x):  # 用于判断所处区间
    for i, a in enumerate(x):
        if a < 0:
            return i

def func_score(nv_df, rf_raw, ind, fund_detail_input, t, back_window, clist, freq_r='w', freq_sp='w'):
    #startTime = time.clock()
    # 先筛选出符合条件的基金类型
    fund_detail = fund_detail_input.copy()
    # 转变时间类型
    fund_detail.ivstTypeInDt = fund_detail.ivstTypeInDt.apply(
        lambda x: to_date(x)[:-2])
    fund_detail.ivstTypeExDt = fund_detail.ivstTypeExDt.apply(
        lambda x: to_date(x)[:-2])
    fund_detail.ivstTypeInDt = pd.to_datetime(fund_detail.ivstTypeInDt)
    fund_detail.ivstTypeExDt = pd.to_datetime(fund_detail.ivstTypeExDt)
    # t_0是回溯期开始往前推6个月
    t_0 = datetime.strptime(t, "%Y-%m-%d") - \
        relativedelta(months=back_window + 6)
    # 先筛选出在列表内的基金
    fund_detail_filt = fund_detail[fund_detail['scndIvstType'].isin(clist)]
    fdf = fund_detail_filt.copy()
    # 'ivstTypeInDt'和 'ivstTypeExDt' 控制了基金处于该时间范围内是某种分类
    # 将NaT格式转为当前时点表示无穷点，筛选出基金进入该分类的时间早于t_0且退出该分类的时间晚于t的基金
    fdf.loc[:, 'ivstTypeExDt'] = fdf['ivstTypeExDt'].replace(
        pd.NaT, datetime.now())
    # fund_detail_filt2 是最终符合标准的基金列表
    fund_detail_filt2 = fdf[(fdf['ivstTypeInDt'] < t_0)
                            & (fdf['ivstTypeExDt'] > datetime.strptime(t, "%Y-%m-%d"))].index.to_list()
    # 下部分为原先的指标计算
    # 调整指数序列为时间序列
    ind.index = pd.to_datetime(ind.index)
    rf = pd.Series(rf_raw.iloc[:, 0].values,
                   index=rf_raw.index, copy=True)  # 无风险利率列表
    #nv_df.index = pd.to_datetime(nv_df.index)
    # 设定回溯期间初始点 = 计算时间节点 - 回溯月份
    ini_t = datetime.strptime(t, "%Y-%m-%d") - \
        relativedelta(months=back_window)
    # 选取基金净值序列在时间窗口内且为上面筛选出的基金
    nv_df_w = nv_df.loc[ini_t.strftime(
        "%Y-%m-%d"):t, fund_detail_filt2] if back_window != 0 else nv_df.loc[:t, fund_detail_filt2]  # 若回溯月份为0则是从头开始
    # 最终输出df格式
    out = pd.DataFrame(index=['年化收益率', '收益率标准差', '最大回撤',
                              'Sharpe比率', 'Calmar比率', 'Alpha', 'Alpha稳定性'], columns=nv_df_w.columns)
    # 指标的收益率计算
    r_ind = (ind/ind.shift(1) - 1).loc[nv_df_w.index, :]
    # 最终结束日期，因为要用于计算真实日期
    end_t = datetime.strptime(t, "%Y-%m-%d")
    # 计算真实经过日期
    delta_t = nv_df_w.apply(lambda x: (
        end_t - x.dropna().index[0]).days + 1 if len(x.dropna()) > 0 else 0, axis=0)
    anufreq_dic = {'d': 250, 'w': 52, 'm': 12, 'y': 1}  # 频率换算字典，用于年化
    r_w = (nv_df/nv_df.shift(1) -
           1).loc[nv_df_w.index, nv_df_w.columns]  # 日收益率列表
    nv_r_w = (nv_df_w.resample(freq_r).last(
    )/nv_df_w.resample(freq_r).last().shift(1) - 1) if freq_r != 'd' else r_w  # 用于标准差计算的nv
    nv_rs_w = (nv_df_w.resample(freq_r).last(
    )/nv_df_w.resample(freq_r).last().shift(1) - 1) if freq_sp != 'd' else r_w  # 用于sharpe计算的nv
    std = nv_r_w.apply(lambda x: x.dropna().std() *
                       ((anufreq_dic[freq_r]) ** (1/2)), axis=0)  # 转化为年化
    spr_std = nv_rs_w.apply(lambda x: x.dropna().std()
                            * ((anufreq_dic[freq_sp]) ** (1/2)), axis=0)  # 转化为年化
    rfm = nv_df_w.apply(
        lambda x: rf[x.dropna().index]).mean()/100  # 对应时间内无风险利率均值
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


def func_rank(input_data):
    # 设定排名，最大回撤是负值，所以是倒序排列，calmar乘以-1所以正序排列。
    rank_data = input_data.rank(ascending=False, axis=1)
    #分数 = Σ排名*权重
    score = rank_data.apply(lambda x: x['年化收益率'] * 0.2 + x['Sharpe比率']
                            * 0.3 + x['Alpha'] * 0.3 + x['Alpha稳定性'] * 0.2, axis=0)
    # 分数越小排名越高
    rank = score.sort_values()
    return rank.iloc[:10].index.to_list()

# 变量分别为：基金净值序列，无风险利率序列，指数序列，基金详情序列，回溯窗口，选择的基金种类列表标准差参数，夏普比参数，起始日，终结日


def func_port(nv_df, rf_raw, ind, fund_detail_input, back_window, clist, freq_r, freq_sp, s_d, e_d):
    # 设一个从s_d到e_d之间的季度最后一天数值，该天数序列从nv_df的索引获得，这样得到的每季度末是交易日
    longdate = pd.Series(nv_df.index)
    longdate.index = longdate
    #如果给定的e_d太迟，则用nv_df最后一天作为e_d，防止给出超过时间序列的配仓
    e_d = nv_df.index[-1] if datetime.strptime(e_d, "%Y-%m-%d") > nv_df.index[-1] else e_d
    q_last = longdate.resample('q').last().loc[s_d:e_d].to_list()
    out = pd.DataFrame(columns=['日期', '代码', '权重'])
    # 每季末回溯过去三年得到一个前十名的排名
    for q in q_last:
        code_list = func_rank(func_score(nv_df, rf_raw, ind, fund_detail_input, datetime.strftime(
            q, "%Y-%m-%d"), back_window, clist, freq_r, freq_sp))
        l_c = len(code_list)
        for l in range(l_c):
            out = out.append(
                {'日期': q, '代码': code_list[l], '权重': 1/l_c}, ignore_index=True)
    out.set_index('日期', inplace = True, drop = True)
    return out


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

def nv_df_clean(fund_data):
    
    data = fund_data.copy()

    data.index = data.iloc[:, 0].apply(lambda x: to_date(x))  # 对净值表里的日期数据进行格式调整，以便于调为时间戳格式

    data.index = pd.to_datetime(data.index)
    data.drop(columns=data.columns[0], inplace=True)  # 将index设置为日期的时间戳格式
    data.fillna(method='ffill', inplace=True)
    return data

def get_netvalue(data, port_inf, dic, b, h):  # 输入data为净值表，port_inf为月度调仓信息

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
        d.iloc[-1] = d.iloc[-2]  # 每月最后一天仍然保有相同权重
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


'''load_data'''
fund = pd.read_csv('./fundInfo.csv', encoding='gbk', index_col=0)
rf = pd.read_excel('./Rf.xlsx', header=0, sheet_name=0, index_col=0)
index_3 = pd.read_excel('./指数数据.xlsx', header=0, sheet_name=0, index_col=0)
fdata = pd.read_csv('./output.csv', index_col=0)
dic_data = fdata.T.to_dict('list')
all_data = pd.read_csv('./navAdj.csv')

#
all_data = nv_df_clean(all_data)
port_information = func_port(all_data, rf, index_3, fund, 36, ['普通股票型基金'], 'm', 'm', '2013-01-01', '2020-03-31')
#port_information = pd.read_csv('./qz.csv', index_col=0)
daily, dailyf, weekly, monthly, weeklyf, monthlyf, output = get_netvalue(
    all_data, port_information, dic_data, 0.1, 90)
# plt.plot(daily)
# plt.plot(dailyf)
