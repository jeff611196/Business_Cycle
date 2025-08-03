#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 09:55:53 2025

@author: jeff
"""
import tidal as td
import pandas as pd
import numpy as np
import json
import glob
import os
from nutrlink import NutrLink
from typing import Any, Dict, List, Union
from tidal.lake_analyzer import LakeAnalyzer


nl: NutrLink = NutrLink(url="https://dev-api.ddt-dst.cc/nutrients/station")

def query_nutrients_ohlcv(tickers: List[str], start_date: Union[str, None] = None, end_date: Union[str, None] = None, adjusted: bool = True) -> pd.DataFrame:
    adjusted_str = "adj" if adjusted else "d"
    filters = [("coid", "in", tickers)] + (
        [("mdate", ">=", pd.Timestamp(start_date, tz="UTC").to_pydatetime())] if start_date is not None else []) + (
        [("mdate", "<=", pd.Timestamp(end_date, tz="UTC").to_pydatetime())] if end_date is not None else [])

    data = nl.get(
        "tej_stock_twn_aprcd1" if adjusted else "tej_stock_twn_aprcd", 
        columns=["coid", "mdate", f"open_{adjusted_str}", f"high_{adjusted_str}", f"low_{adjusted_str}", f"close_{adjusted_str}", "volume"],
        filters=filters
    )
    data['volume'] = data['volume'] * 1000
    data = data.rename(columns={
        "coid": "instrument", "mdate": "datetime", 
        f"open_{adjusted_str}": "open", f"high_{adjusted_str}": "high", f"low_{adjusted_str}": "low", f"close_{adjusted_str}": "close"
    })
    data['datetime'] = (data['datetime'] - pd.Timedelta(hours=8)).dt.tz_convert("Asia/Taipei")

    data = data.set_index(["instrument", "datetime"]).sort_index()

    return data

# 設置回測參數
start_date = '2015-05-01'
end_date = '2025-04-30'

# 設定資料夾路徑
folder_path = '/home/jovyan/business/Textile_signal_lights/general/'

# 取得所有 .csv 檔案的完整路徑
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# 用檔名（不含副檔名）當作 key，把各檔案的內容讀成 DataFrame 存入 dict
industry_daily_score = {
    os.path.splitext(os.path.basename(file))[0]: pd.read_csv(file, index_col=0)
    for file in csv_files
}

for industry, df in industry_daily_score.items():
    df.index = pd.to_datetime(df.index)

folder_path_2 = '/home/jovyan/business/top_k/general/'

csv_files_2 = glob.glob(os.path.join(folder_path_2, '*.csv'))

industry_map = {
    os.path.splitext(os.path.basename(file))[0]: pd.read_csv(file)
    for file in csv_files_2
}

# 1. 讀取股票 OHLCV 數據
print("讀取 OHLCV 數據...")
all_coids = set()
for df in industry_map.values():
    all_coids.update(df['coid'].astype(str).tolist())  # 確保都是字串型別

# 建立 stock_list
stock_list = list(all_coids)
quote_data = query_nutrients_ohlcv(stock_list, start_date, end_date)


coid_to_industry = {}
for industry, df in industry_map.items():
    for coid in df['coid']:
        coid_to_industry[str(coid)] = industry


quote_data = quote_data.reset_index()
quote_data['date'] = pd.to_datetime(quote_data['datetime']).dt.date
quote_data['industry'] = quote_data['instrument'].map(coid_to_industry)

# 加入 industry_score 欄位
def get_industry_score(row):
    industry = row['industry']
    date = pd.to_datetime(row['date']).normalize()  # datetime.date
    if pd.isna(industry):
        return None
    if industry not in industry_daily_score:
        return None
    df = industry_daily_score[industry]
    # 這裡欄位名稱要改成'燈號分數'
    return df.loc[date, '燈號分數'] if date in df.index else None

quote_data['industry_score'] = quote_data.apply(get_industry_score, axis=1)

quote_data['ma60'] = quote_data.groupby('instrument')['close'].transform(lambda x: x.rolling(window=60).mean())
quote_data['low_1y'] = quote_data.groupby('instrument')['low'].transform(lambda x: x.rolling(window=252, min_periods=1).min())

# === 3. 設定 index 給 DSTrader 使用 ===
quote_data = quote_data.set_index(['instrument', 'datetime'])

print("讀取基準數據...")
benchmark_data = query_nutrients_ohlcv(['0050'], start_date, end_date)
# 提取基準數據的 'close' 價格，並設置正確索引
if not benchmark_data.empty:
    benchmark_data = benchmark_data.loc['0050'][['close']]
else:
    print("基準數據 (0050) 為空，無法用於比較。")
    benchmark_data = None # Set to None if benchmark is empty
    
quote_data.head(10)

class YourStrategy(td.BaseStrategy):
    def __init__(self, max_inst, industry_map, industry_daily_score):
        super().__init__()
        self.max_inst = max_inst
        self.industry_map = industry_map               # dict: key=產業名, value=該產業股票DataFrame(coid欄)
        self.industry_daily_score = industry_daily_score   # dict: key=產業名, value=每日燈號分數DataFrame (index=date)
        self.last_sell_price = {}

    def on_trade(self):
        quote = self.quote()  # 取得當日股票行情資料，index是股票代碼

        today = pd.to_datetime(self.datetime.date())   # 取得策略當前日期，注意轉成 Timestamp

        # 計算今天各產業燈號分數 (從industry_daily_score取得當日分數)
        industry_scores_today = {}
        for industry, df_score in self.industry_daily_score.items():
            if today in df_score.index:
                industry_scores_today[industry] = df_score.loc[today, '燈號分數']
            else:
                # 如果今天沒有資料，給一個很大值，避免被選中
                industry_scores_today[industry] = 9999
        
        # 找燈號分數最低的產業(若有多個同分，取第一個)
        selected_industry = min(industry_scores_today, key=industry_scores_today.get)
        # print(f"[{self.datetime}] 選擇產業: {selected_industry}，分數: {industry_scores_today[selected_industry]}")

        # 選出買入 / 賣出候選股票
        buy_candidates = quote[quote['industry_score'] <= 2]
        sell_candidates = quote[quote['industry_score'] >= 4]

        held_stocks = set(self.positions.keys())

        # 限制買入股票必須在選中的產業內
        candidate_stocks = set(self.industry_map[selected_industry]['coid'].astype(str))
        buy_candidates = buy_candidates.loc[buy_candidates.index.isin(candidate_stocks)]

        # ===== 賣出區塊 =====
        for inst in held_stocks:
            if inst in sell_candidates.index:
                current_price = quote.loc[inst, 'close']
                ma60 = quote.loc[inst, 'ma60']
                
                total_quantity = 0
                total_cost = 0
                for pos in self.positions[inst]:
                    total_quantity += pos.quantity
                    total_cost += pos.price * pos.quantity

                if total_quantity > 0:
                    avg_cost = total_cost / total_quantity
                    return_pct = (current_price - avg_cost) / avg_cost

                    if return_pct >= 0.3:
                        self.place_order(inst, -total_quantity, current_price, td.OrderType.MARKET)
                        self.last_sell_price[inst] = current_price
        
        # ===== 買入區塊 =====
        available_cash = self.cash
        current_positions = len(held_stocks - set(sell_candidates.index))
        remaining_slots = self.max_inst - current_positions

        buy_candidates = buy_candidates.sort_values(by='industry_score')
        buy_candidates = buy_candidates.loc[~buy_candidates.index.duplicated(keep='first')]
        buy_list = buy_candidates.loc[~buy_candidates.index.isin(held_stocks)] \
                                  .head(remaining_slots)

        if not buy_list.empty and remaining_slots > 0:
            cash_per_stock = available_cash / len(buy_list)
            for inst in buy_list.index:
                price = buy_list.loc[inst, 'close']
                low_1y = buy_list.loc[inst, 'low_1y']
                if price > 0 :
                    quantity = int(cash_per_stock // price)
                    if quantity > 0:
                        self.place_order(inst, quantity, price, td.OrderType.MARKET)
                        
# Initialize DSTrader object
tidal = td.Tidal(init_cash=10000000, slip_ticks=1, stock_config=td.StockConfig.TW, load_configs=True)

# Add Quote data
tidal.add_quote(quote_data)

# Set strategy object
tidal.set_strategy(YourStrategy(max_inst=20,industry_map=industry_map, industry_daily_score=industry_daily_score))

# Set metric objects
tidal.add_metric(td.metric.AccountInfo())
tidal.add_metric(td.metric.PositionInfo())
tidal.add_metric(td.metric.Portfolio(benchmark_data))

# tidal.metrics['Portfolio'].report

tidal.backtest()

tidal.trade_report

tidal.tdboard()
