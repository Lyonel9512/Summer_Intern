# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 09:19:19 2020

@author: Lyonel
"""

import pandas as pd
import re


def get_infor():  # 调整information信息表里的数据
    for i in range(len(information)):
        if type(information.iloc[i, 0]) == str:
            if information.iloc[i, 0] == '0%\r\n0%\r\n':
                information.iloc[i, 0] = '0%'
            information.iloc[i, 2] = [float(s) for s in re.findall(
                r"\d+\.?\d*", information.iloc[i, 0])]  # 用正则匹配申购信息里的数字
        else:
            information.iloc[i, 2] = [float(information.iloc[i, 0])]
    return information


def get_list_1(raw_list):  # 只有一种情况
    la = []
    lb = []
    for s in range(len(raw_list[2:])):
        if raw_list[2:][s] >= 3:  # 用3作为识别金额和费率的阈值
            if raw_list[2:][s+1] < 3:  # 形式为前一个是金额，后面是费率
                la.append(raw_list[2:][s])  # 金额加入
                if raw_list[2:][s+1] < 3:
                    lb.append(round(raw_list[2:][s+1]/100, 4))  # 费率填入
            else:
                continue
        else:
            continue
    la.append(float('inf'))  # 在最后加入inf
    lb.append(raw_list[1])  # 根据形式，加入超过限额的整数费用

    return [la, lb]


def get_list_2(raw_list):  # 有两种情况
    la = []
    lb = []
    for s in range(len(raw_list[4:])):
        if raw_list[4:][s] >= 3:
            if raw_list[4:][s+1] < 3:
                la.append(raw_list[4:][s])
                lb.append(raw_list[4:][s+1])
            else:
                continue
        else:
            continue
    l1 = []
    l2 = []
    l3 = []
    l4 = []  # l1234 分别作为两种情况填入
    for s in range(len(la)):  # 两种情况依次填入
        if la[s] not in l1:
            l1.append(la[s])
            if lb[la.index(la[s])] < 3:
                l3.append(round(lb[la.index(la[s])]/100, 4))
            la[s] = 0
        else:
            l2.append(la[s])
            if lb[la.index(la[s])] < 3:
                l4.append(round(lb[la.index(la[s])]/100, 4))
            la[s] = 0
    l1.append(float('inf'))
    l2.append(float('inf'))
    l3.append(raw_list[1])
    l4.append(raw_list[3])
    return [[l1, l3], [l2, l4]]


def ol1(raw_list):  # 附加操作
    l1 = []
    l2 = []
    for i in range(len(raw_list)-1):
        if i % 3 == 0:
            l1.append(raw_list[i])
            l2.append(round(raw_list[i+1]/100, 4))
    l1.append(float('inf'))
    l2.append(round(raw_list[-1]/100, 4))
    return [l1, l2]


def split(s):
    key_dict = {'场外': 0, '场外普通投资群体': 1, '场内': 2, '场内普通投资群体': 3,
                '普通投资者': 4, '普通投资群体': 5, '社保,养老金,企业年金': 6, '其他特定投资群体': 7}
    t = [v.split(':') for v in s.split(';')]  # 分行,分为场景+描述
    t = [[v[0].replace('\n', '').replace('\r', ''), v[1]] for v in t]  # 消除\r\n
    keys = []  # 用于存储不同场景
    for v in t:
        if v[0] not in keys:
            keys.append(v[0])
    keys = sorted(keys, key=lambda x: key_dict[x])
    r = [[s[1] for s in t if s[0] == v] for v in keys]  # 不同场景保留不同的描述
    return r, keys


def sp2(s):  # 用于给申购加描述
    s = s.replace(',', ' ')
    t = [v.split(' ') for v in s.split('\r\n')]  # 分行,分为场景+描述
    t = [[v[0].replace('\n', '').replace('\r', ''), v[1]] for v in t[:-1]]
    keys = []  # 用于存储不同场景
    for v in t:
        if v[0] not in keys:
            keys.append(v[0])
    return keys[0]


def get_key_v(_v, k):
    dr = re.compile('[0-9]+')  # 正则表达式
    if ' ' not in _v:  # 表示该描述下没有更多的日期格式供参考
        _v = re.findall(r"\d+\.?\d*\%", _v)  # 尝试匹配数字
        if len(_v) > 0:
            _v = _v[0]
        else:  # 如果不含数字则是 ‘本基金的H类基金份额赎回费率。。。’
            return [0, float('inf')], 0
        r = float(_v[:-1])  # 选择费率
        if r <= 0.00000001:  # 默认为0的费率是超过期限，即inf
            return [float('inf'), float('inf')], r
        elif ('其他特定投资群体' in k) or ('场外' in k) or ('场内' in k) or ('场外普通投资群体' in k)  or ('场内普通投资群体' in k):
            return [float('inf'), float('inf')], r        
        else:
            return [0, 0], float(_v[:-1])  # 0情况表示非受限赎回费率
    a, b = _v.split(' ')  # 将描述分开，日期+费率
    r = float(b[:-1])  # 费率保留百分号前
    if '以下' in a:
        e = int(dr.findall(a)[0])  # 第一个时间段用以下分开
        e = e * 365 if '年' in a else e*30 if '月' in a else e  # 根据本文改变日期单位
        return [1, e], r  # 日期+费率
    elif '以上' in a:
        e = int(dr.findall(a)[0])  # 最后时间段用以上分开
        e = e * 365 if '年' in a else e * 30 if '月' in a else e
        return [e, float('inf')], r
    else:
        e1, e2 = a.split('~')  # 区间段的用~分开：首日+结束日
        e1, e2 = int(dr.findall(e1)[0]), int(dr.findall(e2)[0])  # 匹配为数字
        e1 = e1 * 365 if '年' in a else e1 * 30 if '月' in a else e1  # 根据本文改变日期单位
        e2 = e2 * 365 if '年' in a else e2 * 30 if '月' in a else e2
        return [e1, e2], r


def tran(_v):
    contain_inf = False
    for t in _v:  # 某场景的“日期+费率”结构
        if t[0][1] == float('inf') and t[0][0] != float('inf'):  # 是否为inf表示
            contain_inf = True
    if contain_inf:
        _v = [t for t in _v if not t[0][0] == float('inf')]
    _v = sorted(_v, key=lambda v: v[0])  # 根据日期的大小进行排序
    r1, r2 = [], []
    for t, f in _v:  # 取右端点作为时间点，r费率填入
        r1.append(t[1])
        r2.append(round(f/100, 4))
    return [r1, r2]


def get_offer(information):  # 对information里每个申购费率的字符串做解析
    fee_list = []
    investor_list = []
    for f in range(len(information.iloc[:, 2])):
        if len(information.iloc[f, 2]) > 2:
            if information.iloc[f, 2][0] == information.iloc[f, 2][2]:
                fee_list.append(get_list_2(information.iloc[f, 2]))
                investor_list.append(['普通投资群体', '养老金账户'])
            else:
                fee_list.append(get_list_1(information.iloc[f, 2]))
                investor_list.append(['普通投资群体'])
        elif len(information.iloc[f, 2]) == 1:
            fee_list.append([[float('inf')], [information.iloc[f, 2][0]/100]])
            investor_list.append(['普通投资群体'])
        elif len(information.iloc[f, 2]) == 2:
            fee_list.append([[information.iloc[f, 2][0]],
                             [information.iloc[f, 2][1]]])
            investor_list.append(['普通投资群体'])
    return fee_list, investor_list


def get_list_3(s):
    r, k = split(s)
    r = [[get_key_v(t, k) for t in v] for v in r]
    r = [tran(t) for t in r]
    for v in r:
        if float('inf') not in v[0]:
            v[0].append(float('inf'))
            v[1].append(0.0)
    return r, k


def get_redeem(information):  # 对information里每个赎回费率的字符串做解析
    fee_list = []
    investor_list = []
    for f in range(len(information.iloc[:, 1])):
        if information.iloc[f, 1] == 0:  # 赎回费率为0
            fee_list.append([[[float('inf')], [0]]])
            investor_list.append(['普通投资群体'])
        else:
            fee_list.append(get_list_3(information.iloc[f, 1])[0])
            investor_list.append(get_list_3(information.iloc[f, 1])[1])
    return fee_list, investor_list


def last(information, detail):
    # 此部分是对格式异常的部分进行修正，分别包括规则不对，特殊的养老金申购和香港投资者
    # 对detail进行合并，并对其中异常部分进行修正
    b = []

    for i in information.iloc[:, 0]:
        try:
            if type(i) == str and i != '0%\r\n':
                b.append(sp2(i))
            elif i == '0%\r\n':
                b.append(0)
            elif i == 0:
                b.append(0)
        except IndexError:
            b.append('error')

    bug = pd.DataFrame({'i': information.iloc[:, 0], 'des': b})

    l1 = []  # 表示不符合规则
    l2 = []  # 表示特殊的养老金方式
    l3 = []  # 表示香港投资者
    l4 = []
    for i in bug.index:
        if type(bug.loc[i, 'des']) == str:
            if '以下' in bug.loc[i, 'des']:
                l1.append(i)
            elif '养老金' in bug.loc[i, 'des']:
                l2.append(i)
            elif '香港' in bug.loc[i, 'des'] or 'H类' in bug.loc[i, 'des']:
                l3.append(i)
            elif '笔' in bug.loc[i, 'des']:
                l4.append(i)
    l4.remove('160417.OF')
    temp = information.loc[l4, '申购费解析']
    l4_c = [[[[float('inf')], [s[0]]], get_list_1(s[1:])] for s in temp]
    detail.loc[l4, 'inv1'] = [['普通投资群体', '养老金账户']]
    detail.loc[l4, 'offer'] = l4_c
    detail.loc[l3, 'inv1'] = [['香港投资者']]
    l2_c = [[[[t[3], t[6], t[9], float('inf')], [round(t[4]/100, 4), round(t[7]/100, 4), round(
        t[10]/100, 4), t[2]]], [[float('inf')], [t[0]]]] for t in information.loc[l2, '申购费解析']]
    l1_c = []
    for i in information.loc[l1, '申购费解析']:
        l1_c.append(ol1(i))
    detail.loc[l2, 'inv1'] = [['普通投资群体', '养老金账户']]
    detail.loc[l2, 'offer'] = l2_c
    detail.loc[l1, 'inv1'] = [['普通投资群体']]
    detail.loc[l1, 'offer'] = [i for i in l1_c]
    for i in range(len(detail)):
        if len(detail.iloc[i, 1]) == 1:
            detail.iloc[i, 1] = detail.iloc[i, 1][0]
        if len(detail.iloc[i, 1][1]) == 1 and detail.iloc[i, 1][0][0] == 0 and detail.iloc[i, 1][1][0] > 0:
            detail.iloc[i, 1][0][0] = float('inf')
    for d in detail.loc[:, 'offer']:
        if len(d) == 2 and type(d[0][-1]) == float and d[0][-1] != float('inf'):
            d[0].append(float('inf'))
            d[1].insert(0, 1.5)
            d[1] = [round(v/100, 4) if v < 3 else v for v in d[1]]
    for d in detail.loc[l3, 'redeem']:
        if len(d) == 2 and type(d[1][0]) == list and d[1][0][0] == 0:
            d[1][1].remove(0)
            d[1][0].remove(0)
        elif len(d) == 2 and d[0][0] == 0:
            d[1].remove(0)
            d[0].remove(0)    
    return detail


information = pd.read_excel(
    './基金费率解析.xlsx', header=0, sheet_name=0, index_col=0)

information = get_infor()
fee_list_1 = get_offer(information)[0]
fee_list_2 = get_redeem(information)[0]
inv_list_1 = get_offer(information)[1]
inv_list_2 = get_redeem(information)[1]

detail = pd.DataFrame({'fund': information.index, 'offer': fee_list_1,
                       'redeem': fee_list_2, 'inv1': inv_list_1, 'inv2': inv_list_2})
detail.set_index('fund', inplace=True)

detail = last(information, detail)
# 对细节处做修改
dic_detail = detail.T.to_dict('list')  # 转为字典
