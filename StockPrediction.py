import csv
import pandas as pd
import matplotlib.pyplot as plt
import os

directory = "Stocks"

stock = "NFLX.csv"
df = pd.read_csv(os.path.join(directory, stock))



start_yr = "2019-01-01"
end_yr = "2020-01-01"

start_mnth = "2019-01-01"
end_mnth = "2019-02-01"

start_wk = "2019-01-01"
end_wk = "2019-01-08"

rolling = 2


df = df[df["Date"] > start_mnth]
df = df[df["Date"] < end_mnth]
df = df.set_index("Date")

ma100 = df.Close.rolling(rolling).mean()

# data = list(zip(ma100))
# file_name = "plot_data.csv"
# with open(file_name, "w", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["pred"])
#     writer.writerows(data)

plt.figure(figsize=(12, 6))
plt.plot(ma100, "r")
plt.show()

# # Splitting Data into Training and Testing
# dataTraining = pd.DataFrame(df["Close"][0:int(len(df) * .70)])
# dataTesting = pd.DataFrame(df["Close"][int(len(df) * .70):int(len(df))])
#
# # print(dataTraining.shape)
# # print(dataTesting.shape)
#
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataTrainingArr = scaler.fit_transform(dataTraining)
# # print(dataTrainingArr)
#
# xTrain = []
# yTrain = []
#
# # 100 days
# for i in range(100, dataTrainingArr.shape[0]):
#     xTrain.append(dataTrainingArr[i-100])
#     yTrain.append(dataTrainingArr[i, 0])
#
# # Convert xTrain and yTrain to array
# xTrain, yTrain = np.array(xTrain), np.array(yTrain)
#
# # Machine Learning Model
# model = Sequential()
# model.add(LSTM(units = 50, activation = "relu", return_sequences=True, input_shape=(xTrain.shape[1], 1)))
# model.add(Dropout(0.2))
#
# model.add(LSTM(units = 60, activation = "relu", return_sequences=True))
# model.add(Dropout(0.3))
#
# model.add(LSTM(units = 80, activation = "relu", return_sequences=True))
# model.add(Dropout(0.4))
#
# model.add(LSTM(units = 120, activation = "relu"))
# model.add(Dropout(0.5))
#
# model.add(Dense(units = 1))
#
# model.summary()
#
# model.compile(optimizer = "adam", loss = "mean_squared_error")
# model.fit(xTrain, yTrain, epochs=50)
#
# # model.save("StockPrediction.h5")
#
#
# # For Checking
#
# # Use the past 100 days for predicting
# pastHundDays = dataTraining.tail(100)
# finalDF = pastHundDays.append(dataTesting, ignore_index = True)
#
# # Apply scaling -> values are between 0 to 1
# inputData = scaler.fit_transform(finalDF)
# print(inputData.shape)
#
# xTest = []
# yTest = []
#
# for i in range(100, inputData.shape[0]):
#     xTest.append(inputData[i-100:i])
#     yTest.append(inputData[i, 0])
#
# # Convert to numpy array
# xTest = np.array(xTest)
# # xTest = np.array(xTest, (xTest.shape[1], xTest[2], 1))
# yTest = np.array(yTest)
#
# print(xTest.shape)
# print(yTest.shape)
#
# # Make Predictions
# yPredicted = model.predict(yTest)
#
# print(yPredicted.shape) # Factor for scaling down
