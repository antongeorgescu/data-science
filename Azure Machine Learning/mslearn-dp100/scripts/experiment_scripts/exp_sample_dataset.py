from azureml.core import Run
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

# Get the experiment run context
run = Run.get_context()

# load the diabetes dataset
data = pd.read_csv('data/TACN-original.csv')

# Identify the metrics associated with the experiment
row_count = (len(data))
run.log('observations', row_count)
run.log('start_date',min(data['Date'].tolist()))
run.log('end_date',max(data['Date'].tolist()))

print('Analyzing {} rows of data'.format(row_count))

# Plot and log the count of diabetic vs non-diabetic patients
X0 = data['Date'].tolist() 
y0 = data['Close'].values.tolist()

fig = plt.figure(figsize=(6,6))
ax = fig.gca()    

ax.set_title('Stock Price Chart') 
ax.set_xlabel('Date') 
ax.set_ylabel('Stock Closing Price (USD)')
ax.plot(X0,y0,color="red")
ax.xaxis.set_major_locator(MaxNLocator(5)) 
plt.show()

run.log_image(name='stock_price_chart', plot=fig)

# Save a sample of the data
os.makedirs('outputs', exist_ok=True)
data.sample(100).to_csv("outputs/TACN-sample.csv", index=False, header=True)

# Complete the run
run.complete()
