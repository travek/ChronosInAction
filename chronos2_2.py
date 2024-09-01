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

 
def read_and_parse_csv(file_path):  
    # Read the CSV file  
    df = pd.read_csv(file_path)  
    return df
    
def create_rolling_windows_with_next(df, N):  
    # Ensure 'CLOSE' column exists  
    if '<CLOSE>' not in df.columns:  
        raise ValueError("The 'CLOSE' column is missing in the CSV file.")  
      
    # Extract the 'CLOSE' column  
    close_prices = df['<CLOSE>'].values
      
    # Create rolling windows  
    rolling_windows_with_next = []  
    for i in range(len(close_prices) - N):  
        window = close_prices[i:i + N]  
        next_value = close_prices[i + N]  
        rolling_windows_with_next.append((np.array(window), next_value))  # Create a tuple of 1-d tensor and next value  
      
    return rolling_windows_with_next  


def main(N):  
    file_path = 'SBER_240501_240828.txt'  # Replace with your CSV file path  
      
    # Read and parse the CSV file  
    df = read_and_parse_csv(file_path)  
      
    # Create rolling windows with the next value  
    rolling_windows_with_next = create_rolling_windows_with_next(df, N)  
    #print(rolling_windows_with_next[1])

    # Print the rolling windows with the next value  
    all_ds=0
    all_diff=0
    all_diff2=0
    result_set={'ds_rate':0, 'all_ds':0, 'all_diff':0, 'all_diff2':0.0, 'iter':0}
    for i, (window, next_value) in enumerate(rolling_windows_with_next):  
        #print(f"Rolling window {i + 1}: {window}, next value: {next_value}")  
        context = torch.tensor(window, dtype=torch.float32)
        #print(f'Context: {context}')
        prediction_length = 1
        forecast = pipeline.predict(context, prediction_length)
        #print(f'Forecast: {forecast}' )
        #forecast_index = range(len(window), len(window) + prediction_length)
        low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
        #print(low, median, high)        
        agg_predicted=(low[0]+median[0]+high[0])/3
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
    

