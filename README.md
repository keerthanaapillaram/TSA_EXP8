# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
```

Read the Ridership dataset
```
data = pd.read_csv('Monthly_Ridership.csv')
```

Focus on the '#Ridership' column

```
ridership_data = data[['Ridership']]
```

Display the shape and the first 10 rows of the dataset
```
print("Shape of the dataset:", ridership_data.shape)
print("First 10 rows of the dataset:")
print(ridership_data.head(10))
```

Plot Original Dataset(#Ridership Data)

```
plt.figure(figsize=(12, 6))
plt.plot(ridership_data['Ridership'], label='Original Ridership Data')
plt.title('Original Ridership Data')
plt.xlabel('Time (Months)')
plt.ylabel('Number of Riders')
plt.legend()
plt.grid()
plt.show()
```
Moving Average Perform rolling average transformation with a window size of 5 and 10

```
rolling_mean_5 = ridership_data['Ridership'].rolling(window=5).mean()
rolling_mean_10 = ridership_data['Ridership'].rolling(window=10).mean()
```
Display the first 10 and 20 vales of rolling means with window sizes 5 and 10 respectively

```
rolling_mean_5.head(10)
rolling_mean_10.head(20)
```
Plot Moving Average
```
plt.figure(figsize=(12, 6))
plt.plot(ridership_data['Ridership'], label='Original Ridership Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)', color='orange')
plt.plot(rolling_mean_10, label='Moving Average (window=10)', color='green')
plt.title('Moving Average of Ridership Data')
plt.xlabel('Time (Months)')
plt.ylabel('Number of Riders')
plt.legend()
plt.grid()
plt.show()
```

Perform data transformation to better fit the model
```
if 'Month' in data.columns:
    data['Month'] = pd.to_datetime(data['Month'])
    data.set_index('Month', inplace=True)
```
```
data = data.resample('MS').mean()
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data[['Ridership']]).flatten(),
    index=data.index,
    name='Ridership'
)
```

Exponential Smoothing
```

scaled_data=scaled_data+1 # multiplicative seasonality cant handle non postive values, yes even zeros
x=int(len(scaled_data)*0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul').fit()

test_predictions_add = model_add.forecast(steps=len(test_data))

ax=train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add","test_data"])
ax.set_title('Visual evaluation')

np.sqrt(mean_squared_error(test_data, test_predictions_add))

np.sqrt(scaled_data.var()),scaled_data.mean()

```

Make predictions for one fourth of the data
```
model = ExponentialSmoothing(data, trend='add', seasonal='mul', seasonal_periods=12).fit()
predictions = model.forecast(steps=int(len(data) / 4))
ax = data.plot(figsize=(10, 6))
predictions.plot(ax=ax)
ax.legend(["Monthly Ridership", "Predicted Ridership"])
ax.set_xlabel('Month')
ax.set_ylabel('Number of Monthly Riders')
ax.set_title('Prediction')
plt.show()

```


### OUTPUT:

Original data:

![image](https://github.com/user-attachments/assets/f44f9d9e-d9f2-47cf-9ff5-e200d783579c)

![image](https://github.com/user-attachments/assets/a479fc22-ef8a-4bc9-b1ca-100f263dec6c)

Moving Average:- (Rolling)

window(5):

![image](https://github.com/user-attachments/assets/8986d1ef-371f-4ed3-86f8-237b1aa30929)

window(10):

![image](https://github.com/user-attachments/assets/7a935c2c-fc23-4312-ae3e-400442605ab9)

plot:

![image](https://github.com/user-attachments/assets/610fc950-17f3-4cfc-855f-e569e46c90a6)

Exponential Smoothing:-

Test:

![image](https://github.com/user-attachments/assets/0a49f1c7-72a7-4385-936c-6251a25ab15e)

Performance: 

![image](https://github.com/user-attachments/assets/bd4484e5-338e-4af6-9cb6-93c0daa0fec1)

Prediction:

![image](https://github.com/user-attachments/assets/7b5e172e-b7b1-4f25-8f6d-29f2914993f9)


### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
