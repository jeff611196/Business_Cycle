#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 15:06:23 2025

@author: jeff
"""

import numpy as np
import pandas as pd    
import json
import itertools  
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  
from datetime import datetime
from nutrlink import NutrLink
nl = NutrLink(url="https://dev-api.ddt-dst.cc/nutrients/station")


def INDUSTRY():

    res = nl.get("tw_industry")
    filtered = res[res['ticker'].str.len() == 4]
    selected_columns = ['ticker', 'twse_ind']
    use_df = filtered[selected_columns]
    use_df = use_df.reset_index(drop=True)
    
    # total_data_1 = pd.read_parquet(
    #     f"{PLUMBER_HOST}tej/stock/twn/aind",
    #     storage_options={
    #         "gcp-token": json.dumps(token)
    #     },
    # )

    # filtered = total_data_1[total_data_1['coid'].str.len() == 4]
    # selected_columns = ['coid', 'tejind4_c']
    # use_df = filtered[selected_columns]
    # use_df = use_df.reset_index(drop=True)

    return use_df

class use_time:
    def __init__(self, start_year, end_year):
        self.start_year = start_year+1
        self.end_year = end_year

    def generate_df(self):
        dates = [datetime(year, 12, 1).strftime('%Y-%m-%d') for year in range(self.start_year, self.end_year + 1)]
        df = pd.DataFrame(dates, columns=['date'])
        
        # months = [3, 6, 9, 12]  # 每季的第一天
        # dates = [
        #     datetime(year, month, 1).strftime('%Y-%m-%d')
        #     for year in range(self.start_year, self.end_year + 1)
        #     for month in months
        # ]
        # df = pd.DataFrame(dates, columns=['date'])
        return df

class REVENUE:
    def __init__(self, time_df):
        self.time = pd.to_datetime(time_df['date'])
        self.time = self.time.dt.tz_localize("UTC")
        self.time_ori = time_df['date'].astype(str)

    def generate_df(self, code):
        
        start_date = self.time.min()
        end_date = self.time.max()

        filters = [
            ("mdate", ">=", start_date),
            ("mdate", "<=", end_date),
            ]

        read_data = nl.get("tej_stock_twn_asale", filters=filters)
        revenue_table = read_data.loc[:, ['coid', 'mdate', code]]
        revenue_filtered = revenue_table.loc[
            (revenue_table['coid'].str.len() == 4)
            ]


        # all_data = []
        # for date in self.time_ori:
        #     read_data = pd.read_parquet(
        #         f"{PLUMBER_HOST}tej/stock/twn/asale",
        #         storage_options={
        #             "gcp-token": json.dumps(token),
        #             "start-date": date,
        #             "end-date": date
        #             },
        #         )

        #     revenue_table_ori = read_data.loc[:, ['coid', 'mdate', 'd0007']]
        #     revenue_filtered_ori = revenue_table_ori[revenue_table_ori['coid'].str.len() == 4]
        #     all_data.append(revenue_filtered_ori)

        # final_df = pd.concat(all_data, ignore_index=True)
        return revenue_filtered
    
class RESERVE():
    def __init__(self, time_df):
        self.time = pd.to_datetime(time_df['date'])
        self.time = self.time.dt.tz_localize("UTC")
        self.time_ori = time_df['date'].astype(str)
        
    def generate_df(self, code):
        
        start_date = self.time.min()
        end_date = self.time.max()
        
        filters = [
            ("mdate", ">=", start_date),
            ("mdate", "<=", end_date),
            ]
        
        res = nl.get("tej_financial_statements_twn_aim1aq", filters=filters)
        reserve_table = res[res['acc_code'] == code]
        reserve_filtered = reserve_table.loc[
            (reserve_table['coid'].str.len() == 4)
            ]
        
        return reserve_filtered
    
class GM():
    def __init__(self, time_df):
        self.time = pd.to_datetime(time_df['date'])
        self.time = self.time.dt.tz_localize("UTC")
        self.time_ori = time_df['date'].astype(str)
        
    def generate_df(self, code):
        
        start_date = self.time.min()
        end_date = self.time.max()
        
        filters = [
            ("mdate", ">=", start_date),
            ("mdate", "<=", end_date),
            ]
        
        res = nl.get("tej_financial_statements_twn_aim1aq", filters=filters)
        reserve_table = res[res['acc_code'] == code]
        reserve_filtered = reserve_table.loc[
            (reserve_table['coid'].str.len() == 4)
            ]
        
        return reserve_filtered
    
class OPM():
    def __init__(self, time_df):
        self.time = pd.to_datetime(time_df['date'])
        self.time = self.time.dt.tz_localize("UTC")
        self.time_ori = time_df['date'].astype(str)
        
    def generate_df(self, code):
        
        start_date = self.time.min()
        end_date = self.time.max()
        
        filters = [
            ("mdate", ">=", start_date),
            ("mdate", "<=", end_date),
            ]
        
        res = nl.get("tej_financial_statements_twn_aim1aq", filters=filters)
        reserve_table = res[res['acc_code'] == code]
        reserve_filtered = reserve_table.loc[
            (reserve_table['coid'].str.len() == 4)
            ]
        
        return reserve_filtered
    
class MONEY():
    def __init__(self, time_df):
        self.time = pd.to_datetime(time_df['date'])
        self.time = self.time.dt.tz_localize("UTC")
        self.time_ori = time_df['date'].astype(str)
        
    def generate_df(self, code):
        
        start_date = self.time.min()
        end_date = self.time.max()
        
        filters = [
            ("mdate", ">=", start_date),
            ("mdate", "<=", end_date),
            ]
        
        res = nl.get("tej_financial_statements_twn_aim1aq", filters=filters)
        reserve_table = res[res['acc_code'] == code]
        reserve_filtered = reserve_table.loc[
            (reserve_table['coid'].str.len() == 4)
            ]
        
        return reserve_filtered

industry = INDUSTRY()
industry['twse_ind'] = industry['twse_ind'].str.strip()
use_industry = '資訊服務業'
use_stock = industry.loc[(industry['twse_ind'] == use_industry)]
use_stock = use_stock.reset_index(drop=True)

#先使用以2011 Q1的前20大市值
start_date = datetime.fromisoformat("2011-03-01 00:00:00+00:00")

filters = [
    ("mdate", "==", start_date)
    ]

read_data = nl.get("tej_financial_statements_twn_aim1a", filters=filters)
market_value_table = read_data.iloc[np.where(read_data['acc_code'] == 'MV')[0],:]
market_value_filtered = market_value_table[market_value_table['coid'].str.len() == 4]

# 轉成字串以防 coid/ticker 資料型別不一致
use_ticker_list = use_stock['ticker'].astype(str).tolist()

# 篩出在 use_stock 中的公司
market_value_selected = market_value_filtered[
    market_value_filtered['coid'].astype(str).isin(use_ticker_list)
].copy()

market_value_selected['coid'] = market_value_selected['coid'].astype(int)

# 紡織纖維
# remove_tickers = [1402, 1434, 1303, 1455, 1710, 1409, 4414]
remove_tickers = []

# 排除不需要的 coid
final_mv = market_value_selected[~market_value_selected['coid'].isin(remove_tickers)].copy()

top_20_mv = final_mv.sort_values(by='acc_value', ascending=False)
top_20_mv = final_mv.sort_values(by='acc_value', ascending=False).head(5)
top_20_mv[['coid']].to_csv('/Users/jeff/Desktop/Business Cycle/top_k/top_5/electronic_distribution.csv', index=False)
top_20_mv = pd.read_csv('/Users/jeff/Desktop/Business Cycle/top_k/top_5/electronic_distribution.csv')

# 假設 top_20_mv 是有 coid 的 DataFrame
coid_list = top_20_mv['coid'].astype(str).tolist()

# 假設你分析的是 2018~2024 年
years = list(range(2011, 2025))
quarters = [1, 2, 3, 4]

# 建立所有 coid × 年 × 季 的組合
all_combinations = list(itertools.product(coid_list, years, quarters))

# 建立 DataFrame
panel_df = pd.DataFrame(all_combinations, columns=['coid', 'year', 'quarter'])

# 指標欄位
indicators = [
    "營收 YoY",
    "應收帳款週轉天數",
    "存貨 YoY",
    "存貨週轉天數",
    "毛利率",
    "營業利益率",
    "ROE",
    "營業現金流",
    "現金比",
    "短期借款占比"
]

# 加上空欄位（先填 NaN）
for col in indicators:
    panel_df[col] = pd.NA
    


#時間
start_time = 2010
end_time = 2024
time = use_time(start_time, end_time)
time_df = time.generate_df()

# -- 年頻時間（只取每年 12 月）
year_dates = [datetime(year, 12, 1) for year in range(start_time, end_time + 1)]
year_df = pd.DataFrame(year_dates, columns=["date"])
year_df["date"] = year_df["date"].dt.strftime("%Y-%m-%d")


#近3月累計營收成長率 r25
revenue_instance = REVENUE(year_df)
revenue_result_df = revenue_instance.generate_df('r25').reset_index(drop=True)

revenue_result_df['year'] = pd.to_datetime(revenue_result_df['mdate']).dt.year
revenue_result_df['month'] = pd.to_datetime(revenue_result_df['mdate']).dt.month
revenue_result_df['quarter'] = ((revenue_result_df['month'] - 1) // 3) + 1

# 只取季底月份：3月, 6月, 9月, 12月
season_end = revenue_result_df[revenue_result_df['month'].isin([3, 6, 9, 12])]

# 建立 key 給季資料做 merge
season_end = season_end[['coid', 'year', 'quarter', 'r25']].rename(columns={'r25': '營收 YoY'})

##營收YOY
panel_df['營收 YoY'] = panel_df.merge(season_end, on=['coid', 'year', 'quarter'], how='left')['營收 YoY_y']

reserve_instance = RESERVE(year_df)
reserve_result_df = reserve_instance.generate_df('0170')

reserve_result_df['year'] = pd.to_datetime(reserve_result_df['mdate']).dt.year
reserve_result_df['month'] = pd.to_datetime(reserve_result_df['mdate']).dt.month
reserve_result_df['quarter'] = ((reserve_result_df['month'] - 1) // 3) + 1

reserve_0170 = reserve_result_df.copy()

# 建立年+季欄位以便排序與可視化
reserve_0170['year_quarter'] = reserve_0170['year'].astype(str) + 'Q' + reserve_0170['quarter'].astype(str)

# 依 coid 與季度排序後計算 YoY（年增率）
reserve_0170['acc_value_yoy'] = (
    reserve_0170.sort_values(['coid', 'year', 'quarter'])
           .groupby('coid')['acc_value']
           .pct_change(periods=4, fill_method=None)  # YoY = 同季前一年
)

# 選擇顯示結果（如要保留其他欄位也可以）
reserve_0170_result = reserve_0170[['coid', 'year', 'quarter', 'acc_value', 'acc_value_yoy']]
reserve_season_end = reserve_0170_result[['coid', 'year', 'quarter', 'acc_value', 'acc_value_yoy']].rename(columns={'acc_value_yoy': '存貨 YoY'})
##存貨YOY
panel_df['存貨 YoY'] = panel_df.merge(reserve_season_end, on=['coid', 'year', 'quarter'], how='left')['存貨 YoY_y']

gm_instance = GM(year_df)
gm_result_df = gm_instance.generate_df('R105')

gm_result_df['year'] = pd.to_datetime(gm_result_df['mdate']).dt.year
gm_result_df['month'] = pd.to_datetime(gm_result_df['mdate']).dt.month
gm_result_df['quarter'] = ((gm_result_df['month'] - 1) // 3) + 1

gm_r105 = gm_result_df.copy()

# 建立年+季欄位以便排序與可視化
gm_r105['year_quarter'] = gm_r105['year'].astype(str) + 'Q' + gm_r105['quarter'].astype(str)
gm_season_end = gm_r105[['coid', 'year', 'quarter', 'acc_value']].rename(columns={'acc_value': '毛利率'})
##毛利率
panel_df['毛利率'] = panel_df.merge(gm_season_end, on=['coid', 'year', 'quarter'], how='left')['毛利率_y']

opm_instance = OPM(year_df)
opm_result_df = opm_instance.generate_df('R106')

opm_result_df['year'] = pd.to_datetime(opm_result_df['mdate']).dt.year
opm_result_df['month'] = pd.to_datetime(opm_result_df['mdate']).dt.month
opm_result_df['quarter'] = ((opm_result_df['month'] - 1) // 3) + 1

opm_r106 = opm_result_df.copy()

# 建立年+季欄位以便排序與可視化
opm_r106['year_quarter'] = opm_r106['year'].astype(str) + 'Q' + opm_r106['quarter'].astype(str)
opm_season_end = opm_r106[['coid', 'year', 'quarter', 'acc_value']].rename(columns={'acc_value': '營業利益率'})
##營業利益率
panel_df['營業利益率'] = panel_df.merge(opm_season_end, on=['coid', 'year', 'quarter'], how='left')['營業利益率_y']

money_instance = MONEY(year_df)
money_result_df = money_instance.generate_df('7210')

money_result_df['year'] = pd.to_datetime(money_result_df['mdate']).dt.year
money_result_df['month'] = pd.to_datetime(money_result_df['mdate']).dt.month
money_result_df['quarter'] = ((money_result_df['month'] - 1) // 3) + 1

money_7210 = money_result_df.copy()

# 建立年+季欄位以便排序與可視化
money_7210['year_quarter'] = money_7210['year'].astype(str) + 'Q' + money_7210['quarter'].astype(str)
money_season_end = money_7210[['coid', 'year', 'quarter', 'acc_value']].rename(columns={'acc_value': '營業現金流'})
##營業現金流
panel_df['營業現金流'] = panel_df.merge(money_season_end, on=['coid', 'year', 'quarter'], how='left')['營業現金流_y']
panel_df['營業現金流'] = panel_df['營業現金流']*1000

panel_cleaned = panel_df.dropna(axis=1, how='all')
panel_cleaned_2 = panel_cleaned.dropna().copy()
# panel_cleaned.to_csv("/Users/jeff/Desktop/Business Cycle/cleaned_output.csv", index=False, encoding='utf-8-sig')

features = ['營收 YoY', '存貨 YoY', '毛利率', '營業利益率', '營業現金流']

# 將 inf 轉成 NaN
panel_cleaned_2.loc[:, features] = panel_cleaned_2.loc[:, features].replace([np.inf, -np.inf], np.nan)

# 移除含有 NaN 的「該筆資料」（即：某個股某季）
panel_cleaned_2 = panel_cleaned_2.dropna(subset=features)

for col in ['毛利率', '營業利益率']:
    panel_cleaned_2 = panel_cleaned_2[
        (panel_cleaned_2[col] > -100) & (panel_cleaned_2[col] < 100)
    ]
    
panel_cleaned_2['營收 YoY'] = panel_cleaned_2['營收 YoY'].clip(lower=-100, upper=300)






panel_cleaned_2['date'] = pd.PeriodIndex.from_fields(
    year=panel_cleaned_2['year'],
    quarter=panel_cleaned_2['quarter'],
    freq='Q'
).to_timestamp()

indicators = ['營收 YoY', '存貨 YoY', '毛利率', '營業利益率', '營業現金流']

def rolling_percentile(group):
    group = group.sort_values('date').reset_index(drop=True)

    coid = group['coid'].iloc[0]
    result = group[['year', 'quarter', 'date']].iloc[12:].copy()
    result['coid'] = coid

    for indicator in indicators:
        scores = []
        for i in range(12, len(group)):
            window = group.iloc[i-12:i]
            current = group.iloc[i][indicator]

            # 計算分位數
            percentile = (window[indicator] < current).mean()
            scores.append(percentile)

        # 對應到 1~5 分數
        bins = [-0.01, 0.2, 0.4, 0.6, 0.8, 1]
        labels = [1, 2, 3, 4, 5]
        result[indicator + '_percentile'] = scores
        result[indicator + '_score'] = pd.cut(scores, bins=bins, labels=labels).astype(int)

    return result

results = []
for coid, group in panel_cleaned_2.groupby('coid'):
    df = rolling_percentile(group)
    df['coid'] = coid  # 手動補回 coid
    results.append(df)

df_out = pd.concat(results).reset_index(drop=True)

features_2 = ['營收 YoY_score', '存貨 YoY_score', '毛利率_score', '營業利益率_score', '營業現金流_score']
X = df_out[features_2]

# PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# 建立主成分名稱
pc_names = [f'PC{i+1}' for i in range(pca.components_.shape[0])]

# 建立 DataFrame 顯示每個主成分的 loading
loadings_df = pd.DataFrame(pca.components_, columns=features, index=pc_names)





# # 加入綜合景氣得分
# df_out.loc[:, '綜合景氣得分'] = X_pca[:, 0]

# # 可選：檢視 PCA 解釋的變異
# explained_variance_ratio = pca.explained_variance_ratio_


# # 先建立一個日期欄位，格式可以是 "YYYY-QX"
# df_out.loc[:, 'Date'] = df_out['year'].astype(str) + '-Q' + df_out['quarter'].astype(str)

# # 選取關鍵欄位，並做寬格式轉換
# df_pivot = df_out.pivot(index='Date', columns='coid', values='綜合景氣得分')

# # 重設索引讓 Date 成為欄位
# df_pivot.reset_index(inplace=True)

# df_pivot['產業綜合景氣分數'] = df_pivot.drop(columns=['Date']).mean(axis=1)




def compute_prosperity_score(df_out, X_pca, pca, n_components=1):
    """
    根據前 n 個主成分計算綜合景氣得分，並輸出寬格式表格。
    
    參數：
        df_out: 原始資料（包含 year, quarter, coid）
        X_pca: PCA 轉換後的資料（通常是 pca.transform(X) 的結果）
        pca: 已擬合的 PCA 模型
        n_components: 要使用幾個主成分計算景氣分數（預設為 1）
    
    回傳：
        df_pivot: 寬格式資料，含每家公司與產業平均的景氣得分
    """
    # 限制最大不能超過主成分數
    n_components = min(n_components, X_pca.shape[1])

    # 取出前 n 個主成分的加權平均（依照解釋變異比例）
    weights = pca.explained_variance_ratio_[:n_components]
    weights = weights / weights.sum()  # 讓權重總和為 1
    weighted_score = (X_pca[:, :n_components] * weights).sum(axis=1)

    # 加入景氣得分欄位
    df_out = df_out.copy()
    df_out['綜合景氣得分'] = weighted_score

    # 建立時間欄位
    df_out['Date'] = df_out['year'].astype(str) + '-Q' + df_out['quarter'].astype(str)

    # 寬格式轉換
    df_pivot = df_out.pivot(index='Date', columns='coid', values='綜合景氣得分')
    df_pivot.reset_index(inplace=True)

    # 加總產業平均景氣分數
    df_pivot['產業綜合景氣分數'] = df_pivot.drop(columns=['Date']).mean(axis=1)

    return df_pivot


df_pivot = compute_prosperity_score(df_out, X_pca, pca, n_components=1)


test = compute_prosperity_score(df_out, X_pca, pca, n_components=1)









# 假設 df_pivot['Date'] 長這樣: '2012-Q1' 格式
quarter_to_month = {"Q1": "03", "Q2": "06", "Q3": "09", "Q4": "12"}

# 轉成月份起始日（Q1 → 03月1日）
df_pivot["base_date"] = df_pivot["Date"].apply(
    lambda x: pd.to_datetime(f"{x[:4]}-{quarter_to_month[x[-2:]]}-01")
)

# 延後兩個月
df_pivot["date_shifted"] = df_pivot["base_date"] + pd.DateOffset(months=2)


quantile_pivot = df_pivot.copy()

# 計算四分位
q20 = df_pivot['產業綜合景氣分數'].quantile(0.2)
q40 = df_pivot['產業綜合景氣分數'].quantile(0.4)
q60 = df_pivot['產業綜合景氣分數'].quantile(0.6)
q80 = df_pivot['產業綜合景氣分數'].quantile(0.8)

# 對應燈號分類邏輯
def assign_five_lights(score):
    if score >= q80:
        return '🔴 非常熱'
    elif score >= q60:
        return '🟠 偏熱'
    elif score >= q40:
        return '🟡 穩定'
    elif score >= q20:
        return '🟢 偏冷'
    else:
        return '🔵 非常冷'

# 新增燈號欄位
quantile_pivot['燈號'] = quantile_pivot['產業綜合景氣分數'].apply(assign_five_lights)

save_data = quantile_pivot[['date_shifted','產業綜合景氣分數','燈號']].copy()

save_data['燈號分數'] = save_data['燈號'].map({
    '🔴 非常熱': 5,
    '🟠 偏熱': 4,
    '🟡 穩定': 3,
    '🟢 偏冷': 2,
    '🔵 非常冷': 1
})

daily_rows = []

for _, row in save_data.iterrows():
    start_date = pd.to_datetime(row['date_shifted'])
    end_date = (start_date + pd.offsets.MonthEnd(0))  # 本月最後一天
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    expanded = pd.DataFrame({
        'date': dates,
        '燈號': row['燈號'],
        '燈號分數': row['燈號分數'],
        'date_shifted': row['date_shifted'],
        '產業綜合景氣分數': row['產業綜合景氣分數']
    })

    daily_rows.append(expanded)

# 4. 合併所有每日資料
daily_data = pd.concat(daily_rows, ignore_index=True)



# 假設 monthly_data 是你提供的 DataFrame，包含 'date_shifted', '產業綜合景氣分數', '燈號'
daily_data['end_date'] = daily_data['date_shifted'].shift(-1) - pd.Timedelta(days=1)
daily_data.loc[daily_data['end_date'].isna(), 'end_date'] = pd.Timestamp('2025-04-30')  # 或 today()

# 建立燈號對應分數
light_score_map = {
    '🔴 非常熱': 5,
    '🟠 偏熱': 4,
    '🟡 穩定': 3,
    '🟢 偏冷': 2,
    '🔵 非常冷': 1
}
daily_data['燈號分數'] = daily_data['燈號'].map(light_score_map)

# 展平為每日資料
all_rows = []
for _, row in daily_data.iterrows():
    daily_dates = pd.date_range(start=row['date_shifted'], end=row['end_date'], freq='D')
    df_daily = pd.DataFrame({
        'date': daily_dates,
        'date_shifted': row['date_shifted'],
        '燈號': row['燈號'],
        '燈號分數': row['燈號分數'],
        '產業綜合景氣分數': row['產業綜合景氣分數']
    })
    all_rows.append(df_daily)

daily_full = pd.concat(all_rows).reset_index(drop=True)

daily_full_save = daily_full[['date','燈號分數']]

daily_full_save.to_csv('/Users/jeff/Desktop/Business Cycle/Textile_signal_lights/remove_extreme_values/electronic_distribution.csv', index=False)















from nutrlink.helper import get_ohlcv
price_df = get_ohlcv(
        nl,
        tickers="M2329",
        start="2012-01-01",
        end="2025-05-31",
        adjusted=True,
    )

price_df = price_df.reset_index()
price_df['datetime'] = pd.to_datetime(price_df['datetime'])  # 確保是 datetime 格式
price_df = price_df.set_index(['ticker', 'datetime']).sort_index()

# 依 ticker 分組再轉換月資料
monthly = (
    price_df.groupby(level='ticker')
            .resample('ME', level='datetime')
            .agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
                })
            .reset_index()
)

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

# 顏色對應表
color_map = {
    '🔴 非常熱': 'red',
    '🟠 偏熱': 'orange',
    '🟡 穩定': 'gold',
    '🟢 偏冷': 'green',
    '🔵 非常冷': 'blue'
}

marker_colors = quantile_pivot['燈號'].map(color_map)

# 建立圖
fig = make_subplots(
    rows=1, cols=1,
    specs=[[{"secondary_y": True}]],
    subplot_titles=("產業綜合景氣分數 vs 月收盤價",)
)

# 加入景氣分數（主 y 軸）+ 燈號標色
fig.add_trace(go.Scatter(
    x=quantile_pivot['date_shifted'],
    y=quantile_pivot['產業綜合景氣分數'],
    mode='lines+markers',
    name='產業綜合景氣分數',
    line=dict(color='black'),
    marker=dict(color=marker_colors, size=10, symbol='circle'),
    hovertemplate='日期: %{x|%Y-%m-%d}<br>景氣分數: %{y:.3f}<br>燈號: %{text}',
    text=quantile_pivot['燈號']
), row=1, col=1, secondary_y=False)

# 加入月收盤價（副 y 軸）
fig.add_trace(go.Scatter(
    x=monthly['datetime'],
    y=monthly['close'],
    mode='lines+markers',
    name='月收盤價',
    line=dict(color='green'),
    hovertemplate='日期: %{x|%Y-%m-%d}<br>收盤價: %{y:.2f} 元'
), row=1, col=1, secondary_y=True)

# Layout 設定
fig.update_layout(
    height=600,
    title_text="產業綜合景氣分數與月收盤價（同圖雙 Y 軸）",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig.update_xaxes(title_text="日期")
fig.update_yaxes(title_text="景氣分數", secondary_y=False)
fig.update_yaxes(title_text="收盤價 (元)", secondary_y=True)

pio.renderers.default = 'browser'
fig.show()
