# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:25:00 2022

@author: water
"""

#%% Functions
# Imports
import os
import sys
import warnings

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

os.chdir("../../..")

import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import csv
import math
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from random import randrange
import mplfinance as mpl

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

from pytorch_forecasting import Baseline, NBeats, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import SMAPE

AV_KEY = 'CCT66O03DJN2487Y'

def write_price_data(interval:str, months_prev:int, AV_KEY=AV_KEY, symbol:str = 'SPY'):
    with open(symbol + '_data.csv', mode='w', newline='') as data_file:
        data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
        for m in range(1, months_prev + 1):
            year = math.ceil(m / 12)
            month = m % 12
            if month == 0:
                month = 12
            CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=' + symbol + '&interval=' + interval + 'min&slice=year' + str(year) + 'month' + str(month) + '&apikey=' + AV_KEY
            with requests.Session() as s:
                download = s.get(CSV_URL)
                decoded_content = download.content.decode('utf-8')
                cr = csv.reader(decoded_content.splitlines(), delimiter=',', lineterminator = '\n')
                list_of_dp = list(cr)
                for row in list_of_dp:
                    data_writer.writerow(row)
            time.sleep(21.)
            
def process_ETF_price_data(ticker:str, SMA_five_min_candles:int, BB_five_min_candles:int, num_std:int, market_hours=True):
    ticker_data_path = 'C:\\Users\\water\\Documents\\Projects\\stock_project\\' + ticker + '_data.csv'
    df = pd.read_csv(ticker_data_path)
    UVXY_df = pd.read_csv(r'C:\Users\water\Documents\Projects\stock_project\UVXY_data.csv')
    
    #Converts dataframe to floats
    cols_to_convert = ['open', 'high', 'low', 'close', 'volume']
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        UVXY_df[col] = pd.to_numeric(UVXY_df[col], errors='coerce')
    
    #Converts from past-present to present-past
    df = df[::-1]
    UVXY_df = UVXY_df[::-1]
    df['typical price'] = (df['low'] + df['close'] + df['high']).div(3).values
    
    df['market_open'] = ''
    df['market_close'] = ''
    
    overnight = 0       #dollar amount of overnight change, from close at 16:00 to open at 09:35
    opening = 0         #price at open
    closing = 0         #price at close
    opening_date = 0
    closing_date = 0
    
    if market_hours == True:
        date_log = int(df.iloc[0][0][8:10])
        temp_tp = np.array([])
        temp_vol = np.array([])
        vwap = []
        close_no_overnight = []
        
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            if row['time'] == 'time':
                df.drop([idx], axis=0, inplace=True)
                continue
            split_el = row['time'].split()
            date = int(split_el[0][8:10])
            hour = int(split_el[1][0:2])
            minute = int(split_el[1][3:5])
            
            # Get rid of pre/post trades
            if hour < 9:
                df.drop([idx], axis=0, inplace=True)
                continue
            elif hour == 9 and minute < 35:
                df.drop([idx], axis=0, inplace=True)
                continue
            elif hour > 16:
                df.drop([idx], axis=0, inplace=True) 
                continue
            elif hour == 16 and minute > 0:
                df.drop([idx], axis=0, inplace=True)
                continue
            
            elif hour == 9 and minute == 35:
                df.loc[idx, 'market_open'] = 'market open'
                opening = df.loc[idx, 'open']
                opening_date = date
            elif hour == 16 and minute == 0:
                df.loc[idx, 'market_close'] = 'market close'
                closing = df.loc[idx, 'close']
                closing_date = date
            
            if opening_date != closing_date and closing_date!= 0:
                overnight = opening - closing
            close_no_overnight.append(row['close'] - overnight)
            
# =============================================================================
#             print(row['time'])
#             print(opening)
#             print(closing)
#             print(overnight)
# =============================================================================
            
            # VWAP Calculation
            if date == date_log:
                temp_tp = np.append(temp_tp, row['typical price'])
                temp_vol = np.append(temp_vol, row['volume'])
                vwap.append(np.sum(temp_tp * temp_vol) / np.sum(temp_vol))
            else:
                temp_tp = np.array([row['typical price']])
                temp_vol = np.array([row['volume']])
                vwap.append(np.sum(temp_tp * temp_vol) / np.sum(temp_vol))
                date_log = date
            
        for idx, row in tqdm(UVXY_df.iterrows(), total=UVXY_df.shape[0]):
            if row['time'] == 'time':
                UVXY_df.drop([idx], axis=0, inplace=True)
                continue
            split_el = row['time'].split()
            date = int(split_el[0][8:10])
            hour = int(split_el[1][0:2])
            minute = int(split_el[1][3:5])
            
            # Get rid of pre/post trades
            if hour < 9:
                UVXY_df.drop([idx], axis=0, inplace=True)
                continue
            elif hour == 9 and minute < 35:
                UVXY_df.drop([idx], axis=0, inplace=True)
                continue
            elif hour > 16:
                UVXY_df.drop([idx], axis=0, inplace=True) 
                continue
            elif hour == 16 and minute > 0:
                UVXY_df.drop([idx], axis=0, inplace=True)
                continue
    
    df['Close No Overnight'] = close_no_overnight
    df['vwap'] = vwap
    df['UVXY typical price'] = (UVXY_df['low'] + UVXY_df['close'] + UVXY_df['high']).div(3).values
    df['UVXY volume'] = UVXY_df['volume'].values
    
    '''
    five_min_candles: How many 5 min candles used in Bollinger Band computation
    15 candle: 15 candle sma added to dataframe
    '''
    df[str(SMA_five_min_candles) + ' candle SMA'] = df['typical price'].rolling(SMA_five_min_candles).mean()
    df[str(BB_five_min_candles) + ' candle SMA'] = df['typical price'].rolling(BB_five_min_candles).mean()
    df[str(BB_five_min_candles) + ' candle STD'] = df['typical price'].rolling(BB_five_min_candles).std()
    df['upper band'] = df[str(BB_five_min_candles) + ' candle SMA'] + num_std * df[str(BB_five_min_candles) + ' candle STD']
    df['lower band'] = df[str(BB_five_min_candles) + ' candle SMA'] - num_std * df[str(BB_five_min_candles) + ' candle STD']  
    
    time_idx_list = []
    
    for i in tqdm(range(df.shape[0])):
        time_idx_list.append(i)
        
    df['time_idx'] = time_idx_list
    df['group_id'] = [0] * df.shape[0]
    
    df = df[BB_five_min_candles:]
    
    test_num = int(df.shape[0] * 0.10)
    train_df = df[0:df.shape[0] - test_num]
    test_df = df[df.shape[0] - test_num:]
    
    return train_df, test_df
    
 # Add SPDR sector ETFs, as well as SPY and QQQ
def process_ticker_price_data(ticker:str, SPDR:list, SMA_five_min_candles:int, BB_five_min_candles:int, num_std:int, market_hours=True):
    ticker_data_path = 'C:\\Users\\water\\Documents\\Projects\\stock_project\\' + ticker + '_data.csv'
    ticker_df = pd.read_csv(ticker_data_path)
    SPY_data_path = 'C:\\Users\\water\\Documents\\Projects\\stock_project\\SPY_data.csv'
    SPY_df = pd.read_csv(SPY_data_path)
    QQQ_data_path = 'C:\\Users\\water\\Documents\\Projects\\stock_project\\QQQ_data.csv'
    QQQ_df = pd.read_csv(QQQ_data_path)
    UVXY_data_path = 'C:\\Users\\water\\Documents\\Projects\\stock_project\\UVXY_data.csv'
    UVXY_df = pd.read_csv(UVXY_data_path)
    for i, etf in enumerate(SPDR):
        SPDR_paths[i] = 'C:\\Users\\water\\Documents\\Projects\\stock_project\\' + etf + '_data.csv'
    
    #Converts dataframe to floats
    cols_to_convert = ['open', 'high', 'low', 'close', 'volume']
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        UVXY_df[col] = pd.to_numeric(UVXY_df[col], errors='coerce')
    
    #Converts from past-present to present-past
    df = df[::-1]
    UVXY_df = UVXY_df[::-1]
    df['typical price'] = (df['low'] + df['close'] + df['high']).div(3).values
    
    df['market_open'] = ''
    df['market_close'] = ''
    
    if market_hours == True:
        date_log = int(df.iloc[0][0][8:10])
        temp_tp = np.array([])
        temp_vol = np.array([])
        vwap = []
        
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            if row['time'] == 'time':
                df.drop([idx], axis=0, inplace=True)
                continue
            split_el = row['time'].split()
            date = int(split_el[0][8:10])
            hour = int(split_el[1][0:2])
            minute = int(split_el[1][3:5])
            
            # Get rid of pre/post trades
            if hour < 9:
                df.drop([idx], axis=0, inplace=True)
                continue
            elif hour == 9 and minute < 35:
                df.drop([idx], axis=0, inplace=True)
                continue
            elif hour > 16:
                df.drop([idx], axis=0, inplace=True) 
                continue
            elif hour == 16 and minute > 0:
                df.drop([idx], axis=0, inplace=True)
                continue
            
            elif hour == 9 and minute == 35:
                df.loc[idx, 'market_open'] = 'market open'
            elif hour == 16 and minute == 0:
                df.loc[idx, 'market_close'] = 'market close'
            
            # VWAP Calculation
            if date == date_log:
                temp_tp = np.append(temp_tp, row['typical price'])
                temp_vol = np.append(temp_vol, row['volume'])
                vwap.append(np.sum(temp_tp * temp_vol) / np.sum(temp_vol))
            else:
                temp_tp = np.array([row['typical price']])
                temp_vol = np.array([row['volume']])
                vwap.append(np.sum(temp_tp * temp_vol) / np.sum(temp_vol))
                date_log = date
            
        for idx, row in tqdm(UVXY_df.iterrows(), total=UVXY_df.shape[0]):
            if row['time'] == 'time':
                UVXY_df.drop([idx], axis=0, inplace=True)
                continue
            split_el = row['time'].split()
            date = int(split_el[0][8:10])
            hour = int(split_el[1][0:2])
            minute = int(split_el[1][3:5])
            
            # Get rid of pre/post trades
            if hour < 9:
                UVXY_df.drop([idx], axis=0, inplace=True)
                continue
            elif hour == 9 and minute < 35:
                UVXY_df.drop([idx], axis=0, inplace=True)
                continue
            elif hour > 16:
                UVXY_df.drop([idx], axis=0, inplace=True) 
                continue
            elif hour == 16 and minute > 0:
                UVXY_df.drop([idx], axis=0, inplace=True)
                continue
        
    df['vwap'] = vwap
    df['UVXY typical price'] = (UVXY_df['low'] + UVXY_df['close'] + UVXY_df['high']).div(3).values
    df['UVXY volume'] = UVXY_df['volume'].values
    
    '''
    five_min_candles: How many 5 min candles used in Bollinger Band computation
    15 candle: 15 candle sma added to dataframe
    '''
    df[str(SMA_five_min_candles) + ' candle SMA'] = df['typical price'].rolling(15).mean()
    df[str(BB_five_min_candles) + ' candle SMA'] = df['typical price'].rolling(BB_five_min_candles).mean()
    df[str(BB_five_min_candles) + ' candle STD'] = df['typical price'].rolling(BB_five_min_candles).std()
    df['upper band'] = df[str(BB_five_min_candles) + ' candle SMA'] + num_std * df[str(BB_five_min_candles) + ' candle STD']
    df['lower band'] = df[str(BB_five_min_candles) + ' candle SMA'] - num_std * df[str(BB_five_min_candles) + ' candle STD']  
    
    time_idx_list = []
    
    for i in tqdm(range(df.shape[0])):
        time_idx_list.append(i)
        
    df['time_idx'] = time_idx_list
    df['group_id'] = [0] * df.shape[0]
    
    df = df[BB_five_min_candles:]
    
    test_num = int(df.shape[0] * 0.10)
    train_df = df[0:df.shape[0] - test_num]
    test_df = df[df.shape[0] - test_num:]
    
    return train_df, test_df

def plot_df_days(df, starting_candle:int, SMA_five_min_candles, days=5):

    close_lines = df['close'].iloc[starting_candle:starting_candle * (days + 1) + 1]
    vwap_lines = df['vwap'].iloc[starting_candle:starting_candle * (days + 1) + 1]
    close_no_overnight = df['Close No Overnight'].iloc[starting_candle:starting_candle * (days + 1) + 1]
    upper_band_lines = df['upper band'].iloc[starting_candle:starting_candle * (days + 1) + 1]
    lower_band_lines = df['lower band'].iloc[starting_candle:starting_candle * (days + 1) + 1]
    sma_lines = df[str(SMA_five_min_candles) + ' candle SMA'].iloc[starting_candle:starting_candle * (days + 1) + 1]
    index_list = [i for i in range(starting_candle, starting_candle * (days + 1) + 1)]
    
    plt.plot(index_list, close_lines)
    plt.plot(index_list, vwap_lines)
    plt.plot(index_list, sma_lines)
    plt.plot(index_list, upper_band_lines)
    plt.plot(index_list, lower_band_lines)
    plt.plot(index_list, close_no_overnight)
    
    plt.show()
    
#Candle period in minutes
def SPY_trainer(data, max_prediction_length:int, max_encoder_length:int, batch_size:int, SMA_length:int, candle_period:int, logger_path:str='stonk_net_logs', early_stopping:str=True):
    logger_file_name = 'SPY' + str(candle_period) + 'min' + str(SMA_length) + 'SMA'
    
    training_cutoff = data['time_idx'].max() - max_prediction_length
    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx = 'time_idx',
        target = 'Close No Overnight',
        group_ids=['group_id'],
        min_encoder_length = 0,
        max_encoder_length = max_encoder_length,
        max_prediction_length = max_prediction_length,
        static_categoricals = [],
        static_reals = [],
        time_varying_known_categoricals = ['market_open', 'market_close'],
        variable_groups = {},
        time_varying_known_reals = ['time_idx'],
        time_varying_unknown_categoricals = [],
        time_varying_unknown_reals = [
            'open',
            'high',
            'low',
            'close',
            'typical price',
            'volume',
            'Close No Overnight',
            'vwap',
            'UVXY typical price',
            'UVXY volume',

            '60 candle SMA',
            '60 candle STD',
            'upper band',
            'lower band'
        ],
        add_relative_time_idx = True,
        add_target_scales = True,
        add_encoder_length = True
    )
    
    #Create Validation Set
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
    
    batch_size = 64
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
    
    #Baseline absolute mean error
    actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
    baseline_predictions = Baseline().predict(val_dataloader)
    (actuals - baseline_predictions).abs().mean().item()
    
    #Configure network and trainer
    pl.seed_everything(42)
    trainer = pl.Trainer(gpus=1, gradient_clip_val=0.1)
    
    hidden_size = int(2 * max_encoder_length / 3)
    hidden_continuous_size = int(max_encoder_length / 3)
    
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=hidden_size,
        attention_head_size=1, #1
        dropout=0.1,
        hidden_continuous_size=hidden_continuous_size,
        output_size=9,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=4
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
    
    #Find optimal learning rate
    res = trainer.tuner.lr_find(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=10.0,
        min_lr = 1e-6
    )
    
    #configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger(save_dir=logger_path, name=logger_file_name)  # logging results to a tensorboard
    if early_stopping == True:
        callbacks=[lr_logger, early_stop_callback]
    else:
        callbacks = [lr_logger]
    
    trainer = pl.Trainer(
        max_epochs=100,
        gpus=1,
        weights_summary="top",
        gradient_clip_val=0.1,
        limit_train_batches=30,  # coment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=callbacks,
        logger=logger,
    )
    
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=hidden_size,
        attention_head_size=1, #1
        dropout=0.1,
        hidden_continuous_size=hidden_continuous_size,
        output_size=9,  # 7 quantiles by default
        loss=QuantileLoss(),
        log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
    
    # fit network
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    raw_predictions = best_tft.predict(val_dataloader, mode='raw', return_x=False)

    return raw_predictions['prediction']

def nbeats_trainer(data):
    max_encoder_length = 60
    max_prediction_length = 20
    
    training_cutoff = data["time_idx"].max() - max_prediction_length
    
    context_length = max_encoder_length
    prediction_length = max_prediction_length
    
    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="typical price",
        categorical_encoders={"group_id": NaNLabelEncoder().fit(data.group_id)},
        group_ids=["group_id"],
        # only unknown variable is "value" - and N-Beats can also not take any additional variables
        time_varying_unknown_reals=['typical price'],
        max_encoder_length=context_length,
        max_prediction_length=prediction_length,
    )

    validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training_cutoff + 1)
    batch_size = 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    
    # calculate baseline absolute error
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    baseline_predictions = Baseline().predict(val_dataloader)
    SMAPE()(baseline_predictions, actuals)
    
    pl.seed_everything(42)
    trainer = pl.Trainer(gpus=1, gradient_clip_val=0.01)
    net = NBeats.from_dataset(training, learning_rate=3e-2, weight_decay=1e-2, widths=[32, 512], backcast_loss_ratio=0.1)
    
    # find optimal learning rate
    res = trainer.tuner.lr_find(net, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, min_lr=1e-5)
    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    fig.show()
    net.hparams.learning_rate = res.suggestion()
    
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    trainer = pl.Trainer(
        max_epochs=100,
        gpus=1,
        weights_summary="top",
        gradient_clip_val=0.01,
        callbacks=[],
        limit_train_batches=30,
    )
    
    
    net = NBeats.from_dataset(
        training,
        learning_rate=4e-3,
        log_interval=10,
        log_val_interval=1,
        weight_decay=1e-2,
        widths=[32, 512],
        backcast_loss_ratio=1.0,
    )
    
    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = NBeats.load_from_checkpoint(best_model_path)
                                             
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    predictions = best_model.predict(val_dataloader)
    (actuals - predictions).abs().mean()
    
    raw_predictions, x = best_model.predict(val_dataloader, mode="raw", return_x=True)
    
    best_model.plot_prediction(x, raw_predictions, idx=0, add_loss_to_title=True);
        
    return raw_predictions['prediction']

def plot_quant_prediction(test_df, prediction, sample_idx):         #SINGULAR
    max_slopes = 3
    pred = prediction[0, :, 4]
    print(pred)
    actual = test_df.iloc[sample_idx-20:sample_idx+12]
    x = [i for i in range(12)]
    slopes = [pred[i] / (i + 1) for i in range(max_slopes)]
    max_slope = max(slopes)
    min_slope = min(slopes)
    
    plt.figure()
    plt.plot([i for i in range(-20, 12, 1)], actual['close'])
    plt.plot([i for i in range(-20, 12, 1)], actual['5 candle SMA'])
    plt.plot(x, pred)
    #plt.axvline(x=0, color='red', linestyle='--')
    plt.axvline(x=3, color='red', linestyle='--')
    plt.grid(color='k')
    plt.fill_between(x, y1=prediction[0, :, 3], y2=prediction[0, :, 5], facecolor='orange', alpha=0.3)
    plt.legend(['Actual Price', '5 candle SMA', 'Prediction'])
    plt.title(str(actual.iloc[20]['time']))
# =============================================================================
#     plt.text(max_slope)
#     plt.text(min_slope)
# =============================================================================
    
    plt.show()

# Mode: 'sample' tests 20 data points, 'all' tests all data points. Output quant: (dp, 12, 9)
def test_predictions(
        trainer:str,
        train_data, 
        test_data, 
        logger_path:str, 
        max_prediction_length:int,
        max_encoder_length:int,
        batch_size:int,
        SMA_length:int,
        candle_period:int,
        early_stopping:str=True,
        test_mode:str='sample'
    ):
    
    random.seed(42)
    if test_mode == 'all':
        sample_length = len(test_data)
        sample_indices = [i for i in range(70, sample_length, 3)]
    elif test_mode == 'sample':
        sample_length = 20
        sample_indices = [randrange(0, len(test_data)) for i in range(sample_length)]

    quant_tensor = torch.zeros(sample_length, 12, 9)
    for i, idx in enumerate(sample_indices):
        print(i)
        data = pd.concat([train_data.iloc[idx:], test_data[:idx]], ignore_index=True)
        if trainer == 'SPY_trainer':
            quant = SPY_trainer(data, 
                                max_prediction_length=max_prediction_length, 
                                max_encoder_length=max_encoder_length, 
                                batch_size=batch_size, 
                                SMA_length=SMA_length, 
                                candle_period=candle_period, 
                                early_stopping=early_stopping
                    )
            plot_quant_prediction(test_df=test_data, prediction=quant, sample_idx=idx)
            quant_tensor[i, :, :] = quant
        elif trainer == 'nbeats_trainer':
            quant = nbeats_trainer(data)
        print(quant)
        
    return quant_tensor, sample_indices

def plot_quant_predictions(test_df, quant_tensor, sample_indices):
    assert quant_tensor.shape[0] == len(sample_indices), 'Quant tensor and sample indices dims are different'
    for i in range(quant_tensor.shape[0]):
        idx = sample_indices[i]
        print(idx)
        pred = quant_tensor[i, :, 4]
        actual = test_df.iloc[idx-20:idx+12]
        print(pred)
        
        plt.figure()
        plt.plot([i for i in range(-20, 12, 1)], actual['typical price'])
        plt.plot([i for i in range(12)], pred)
        
        plt.show()
    
# Process quantiles (expecting 9)
# Price delta: delta to set price targets

# =============================================================================
# def process_quantiles(price_delta, quant_tensor):
#     upper_target = typical_price + price_delta
#     lower_target = typical_price + price_delta
#     return
# =============================================================================
    
#%% Build data
SPY_train_df, SPY_test_df = process_ETF_price_data('SPY', SMA_five_min_candles=5, BB_five_min_candles=60, num_std=2)
plot_df_days(SPY_train_df, starting_candle=10000, SMA_five_min_candles=5, days=2)
#SPY_trainer(SPY_train_df, max_prediction_length=12, max_encoder_length=700, batch_size=64, SMA_length=15, candle_period=5, early_stopping=False)
#%% Train
quant_tensor, sample_indices = test_predictions(trainer='SPY_trainer',
                                                train_data=SPY_train_df, 
                                                test_data=SPY_test_df, 
                                                logger_path='stonk_net_logs', 
                                                max_prediction_length=12, 
                                                max_encoder_length=24, 
                                                batch_size=64, 
                                                SMA_length=5, 
                                                candle_period=5,
                                                early_stopping=False,
                                                test_mode='all'
                                                )
#plot_quant_predictions(SPY_test_df, quant_tensor, sample_indices)