import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
import math

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

pipeline = ChronosPipeline.from_pretrained(
  "amazon/chronos-t5-base",
  device_map="cpu",
  torch_dtype=torch.float32,
)

class rates(object):
    def __init__(self, date, time, open_, high, low, close, vol=0.0):
        self.date=date
        self.time=time
        self.o=open_
        self.h=high
        self.l=low
        self.c=close
        self.vol=vol
        self.vwap=0.0
        self.ohlc=(open_+high+low+close)/4.0
        self.hlc=(high+low+close)/3.0


 

    



def read_rates():

    file_names=["SBER_240501_240828.txt"]
    v_rates=list()
    for fn in file_names:
        read_rates=open(fn, 'r') 
        print("Start reading rates from file %s..." % (fn))

        for line in read_rates:
            l1=line.split(',')
            if (l1[2].isdigit() and int(l1[3])<=184000 and int(l1[3])>=100000):    #
                rts=rates(int(l1[2]),int(l1[3]), float(l1[4]), float(l1[5]), float(l1[6]), float(l1[7]), float(l1[8]) )
                v_rates.append(rts)
        read_rates.close()  
    
    print("Reading rates from file is completed!")
    return v_rates

def create_working_data(v_rates, N):
    rolling_windows_with_next = []
    window=[]
    for i in range(len(v_rates) - N):
        window=[]
        if v_rates[i + N-1].date==v_rates[i].date and v_rates[i + N].date==v_rates[i].date:
            for j in range(i, i + N):
                window.append(v_rates[j].c)
            #window = close_prices[i:i + N]
            #print(window)
            average=np.mean(np.array(window))
            window2=np.array(window)-average
            next_value = v_rates[i + N].c
            rolling_windows_with_next.append((np.array(window), next_value, average, window2))  # Create a tuple of 1-d tensor and next value          
    return rolling_windows_with_next



def main(N):  
    #file_path = 'SBER_240501_240828.txt'  # Replace with your CSV file path  
    v_rates=read_rates()
    rolling_windows_with_next=create_working_data(v_rates, N)    

    # Print the rolling windows with the next value  
    all_ds=0
    all_diff=0
    all_diff2=0
    result_set={'ds_rate':0, 'all_ds':0, 'all_diff':0, 'all_diff2':0.0, 'iter':0}
    for i, (window, next_value, average, window2) in enumerate(rolling_windows_with_next):  
        #print(f'window: {window}, next: {next_value}, average: {average}, window2: {window2}')
        #context = torch.tensor(window, dtype=torch.float32)
        context2 = torch.tensor(window2, dtype=torch.float32)
        #print(f'Context: {context}')
        prediction_length = 1
        #forecast = pipeline.predict(context, prediction_length)
        forecast2 = pipeline.predict(context2, prediction_length)
        #print(f'Forecast: {forecast}' )
        #forecast_index = range(len(window), len(window) + prediction_length)
        #low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
        #low, median, high = np.quantile(forecast2[0].numpy(), [0.1, 0.5, 0.9], axis=0)
        q=forecast2[0].numpy()
        s=np.sum(q,axis=0)
        l=len(q)
        #print(low, median, high)        
        agg_predicted=s[0]/l #(low[0]+median[0]+high[0])/3
        agg_predicted=agg_predicted+average
        
        ds=np.sign((next_value - window[-1])*(agg_predicted - window[-1]))
        all_ds=all_ds+ds
        diff=math.sqrt((next_value - agg_predicted)*(next_value - agg_predicted))
        all_diff=all_diff+diff
        diff2=1.0*ds*abs((next_value - window[-1])/window[-1]*100)
        all_diff2=all_diff2+diff2
        #print(f'DS: {ds}, Next: {next_value}, Current: {window[-1]}, predicted: {agg_predicted:.2f}, diff: {diff2:.2f}, all_diff2: {all_diff2:.2f}')
        print(f' Inter: {i}, ds_rate: {all_ds/(i+1)*100:.2f},  DS_ALL: {all_ds:.2f}, DS: {ds}, Predicted: {agg_predicted:.2f}, actual: {next_value}, diff: {all_diff/(i+1):.4f}, diff2: {all_diff2:.2f}')
        result_set["ds_rate"]=all_ds/(i+1)*100.0
        result_set["all_ds"]=all_ds/(i+1)*100.0
        result_set["all_diff"]=1.0*all_diff/(i+1)
        result_set["all_diff2"]=1.0*all_diff2
        #if i>2:
        #    return result_set
        
    return result_set
  
if __name__ == "__main__":  
    
    for N in range(5, 16):
        #N = 7  # Replace with your desired window length  }  
        result_set=main(N)
        with open('Chronos_results_base.txt', 'a') as file:	
            file.write(f'N: {N}, ds_rate: {result_set["ds_rate"]:.2f}, ds: {result_set["all_ds"]:.2f}, all_diff: {result_set["all_diff"]:.4f}, all_diff2: {result_set["all_diff2"]}  \n')    
    

