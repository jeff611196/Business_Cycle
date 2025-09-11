## 快速開始

1. clone 專案：
   ```bash
   git clone https://gitlab.com/dst-dev/business-cycle.git
   ```
2. 進入專案資料夾：
   ```bash
   cd business-cycle
   ```
3. 建立 conda 環境：
   ```bash
   conda create -p ./env python=3.10
   ```
4. 使用建立的 conda 環境：
   ```bash
   conda activate ./env
   ```
5. 取得 gcp token：
   ```bash
   gcloud auth application-default login
   ```
6. 安裝依賴套件：
   ```bash
   cat requirements.txt | xargs -n 1 -L 1 pip install
   ```

## 專案說明

本專案主要針對台灣產業進行分組、景氣燈號計算及回測交易策略設計，流程涵蓋財報處理、產業分數計算與策略驗證


## 資料夾結構與說明
### 產業分組規則

依照長期指數走勢區分為兩大類：

- 長多趨勢(性質較接近的產業分在同一類)

  - my_use: semiconductor、online、information、electronic_distribution

  - my_use_2: food、electronic_components、electrical_machine、electrical_cables

- 盤整趨勢

  - group_1：傳統製造與原物料工業

      rubber、paper、glass、cement、building_materials

  - group_2：加工製造與科技應用產業

      textile_fibers、optoelectronics、chemical、car

  - group_3：服務業與高技術導向產業

      trade_department、tourism、biotechnology

註：所有產業均可於 nl.get("tw_industry") 中取得

## 資料夾內容

- pca/  
儲存產業分組並使用 PCA (1D～3D) 進行回測結果

- buy_and_hold_condition/  
回測結果:  
condition_2 至 condition_5 對應 main.py 中 YourStrategy class 的條件設定  
return_greater_50_quarterly_line：報酬率至少 50% 且股價跌破季線才賣出

- topk/(由於每年市值排序不同，個股有人工挑選過，若刪除資料後續回測績效將無法復現)  
依照產業分組選定個股(例:以市值取TOP5)  

- Textile_signal_lights/  
使用 topk 選定的個股財報計算產業景氣燈號  

   - 財報極端值處理：  
      毛利率、營業利益率：-100% ~ 100%  
      營收 YoY：-100% ~ 300%

   - remove_extreme_values_my：加入極端值條件後的完整產業燈號
   
   - 資料夾名稱後方數字 (2、3) 表示使用的 PCA 維度，未標示則為一維

- goodinfo/  
儲存 2011 年以前財報數據
   - 產業分組 → 個股 → 財報三表 (資產負債表、損益表、現金流量表)
   - 僅 my_use 分組有完整財報抓取

## 執行流程

1. goodinfo.ipynb

   財報三表處理與合併：

      - Input：資產負債表、損益表、現金流量表

      - Output：合併後的 Excel 檔

2. slide_window.ipynb

   產出指定產業的個股(例:半導體前五大市值)及其景氣分數

   - input:  
      市值(tej_financial_statements_twn_aim1a、acc_code = MV)  
      營收(tej_stock_twn_asale)  
      存貨(tej_financial_statements_twn_aim1aq、acc_code = 0170)  
      營業毛利率(tej_financial_statements_twn_aim1aq、acc_code = R105)  
      營業利益率(tej_financial_statements_twn_aim1aq、acc_code = R106)  
      營業現金流(tej_financial_statements_twn_aim1aq、acc_code = 7210)

   - output:  
      產業景氣分數

注意：最後一行 fig.show() 第一次執行需手動停止並再次執行，否則會持續運行

3. main.ipynb

   根據slide_window的產出設計交易策略並回測  
   (YourStrategy中conditions為根據景氣燈號分數所設計的策略)

   策略邏輯：

      - 賣出條件：

         if return_pct >= 0.5 and current_price < ma60  
         (報酬率 ≥ 50% 且股價跌破 60 日均線)

      - 買入條件：

         if price > 0 and low_1y > 0 and price <= 1.3 * low_1y  
         (近一年漲幅 ≤ 30%)

   input:  
      分組的產業分數  
      分組的產業個股

   output:  
      策略累積報酬圖

