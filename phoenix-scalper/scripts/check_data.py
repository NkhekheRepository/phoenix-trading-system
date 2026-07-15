#!/usr/bin/env python3
import pandas as pd, os

root = '/freqtrade/user_data/data/binance/futures'
new = ['SOL','XRP','ADA','AVAX','DOGE','DOT','LINK','APT','SUI','ARB','PEPE','WIF','FET','WLD','UNI','NEAR','FTM','RUNE','GALA','IMX','STX','SEI','ICP','FIL','BONK']
for p in new:
    f = os.path.join(root, f'{p}_USDT_USDT-5m-futures.feather')
    if os.path.exists(f):
        df = pd.read_feather(f)
        print(f'{p:6s} 5m: {len(df):>6d} candles  {str(df["date"].min())[:10]} to {str(df["date"].max())[:10]}')
    else:
        print(f'{p:6s} 5m: MISSING')
