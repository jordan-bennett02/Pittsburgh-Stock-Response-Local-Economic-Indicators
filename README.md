# Pittsburgh Stock Modeling with Local Economic Data
----------------------------------------------------
- Created a variety of models for time series data forecasting for 12 months into the future for Pittsburgh headquartered corporations in different industries with and without the presence of local economic variables like local unemployment and median home price. A twelve month period into the future is extreme for stock forecasting and generally unadvisable for profit optimization, However, our primary purpose was to discern if our local economic data had measuable effects on model accuracy. We colleted data from January 1st, 2000 until July 31st, 2025 and saved the period July 2024 -July 2025 for testing purposes.
- Opted to compress the data into monthly percent changes due to modeling attribute type. The stock prices were given in dollars ranging from 1 to 200 dollars per stock, while unemployment was given as a percentage and median home prices are in the range of 80 to 250 thousand dollars. The industries that we modeled compromised different sectors of Pittsburgh's private sector economy.

- Models utilized include SARIMAX modeling, tree based models like RandomForestRegression, XGBoost, hybrid SARIMAX tree models and LSTM models for forecasting. We tuned the SARIMAX model family utilizing ACF/PACF plots and utilized gridsearch for tuning tree models.

----------------------------------------------------
# Code and Resources Used
- Python 3.12.6

- Packages Used: numpy, pandas, yfinance, matplotlib, scipy, statsmodels, sklearn, xgboost, pytorch.

----------------------------------------------------
# References
- Stock data was collected via the yfinance module. Local unemployment data was also collected via yfinance from FRED. You can find the data here: https://fred.stlouisfed.org/series/PITT342UR.

- Local median home data was collected via zillow. We opt not to use the seasonally adjusted data in order to seasonally adjust on our own. You can find the data here: https://www.zillow.com/research/data/?msockid=17d3ea0fafc060ec0404f861aed261f0.

-----------------------------------------------------
# Data Cleaning
- All of the data we collected was very clean. Our stock data was given as a daily close during business days, while unemployment and home prices were given monthly.

- Given the varying time and magnitude scale of the data, we opted to reformat each data into a monthly percent change. This has the effect of smoothing some of the noise in the stock data but also induces differencing into our stock data, useful for achieving stationarity for time series modeling.

-----------------------------------------------------
# EDA
- Looking at the raw variables we see large fluctuations during the 2008 financial crisis and COVID 19 pandemic. We also see stock volatility during the 2022 invasion of Ukraine by Russia.

- Both of the monthly unemployment data and median home prices exhibited some yearly seasonality as shown via the small uniform fluctuations in the graph below.
  ![Stocks, Unemployment, Median Home Prices](https://github.com/jordan-bennett02/Pittsburgh-Stock-Response-Local-Economic-Indicators/blob/main/history_COVID_19.png)

- After processing our data to a uniform scale and differencing, we see that the data is relatively normal disregarding a few outliers.
  ![QQ Plots](https://github.com/jordan-bennett02/Pittsburgh-Stock-Response-Local-Economic-Indicators/blob/main/qq_plot_normality.png)
- Before beginning modeling, we utilized the ACF and PACF plots to select parameters for our SARIMAX model family as well as what lag features we should include for our other models.
  ![ACF/PACF Plots for PNC](https://github.com/jordan-bennett02/Pittsburgh-Stock-Response-Local-Economic-Indicators/blob/main/ACF_PACF_Stocks.png)

  ---------------------------------------------------
  # Model Buildling
  - Before any modeling, we applied the Augmented Dickey-Fuller test in order to deduce that our data was more than likely sampled from a stationary process. We also deduced from the ACF and PACF plots the most statistically viable parameters for our SARIMAX family of models. Each of our stock variables were most appropriately modeled only using autoregressive parameters. We utilized these parameters for our first three models: ARIMA, SARIMA and SARIMAX.
 
  - Because of the apparent seasonality in our unemployment and housing price data, we also seasonally differenced each of our stock variables and examined the ACF and PACF plots for these series to determine our seasonal parameters (P,D,Q,s) for the SARIMA and SARIMAX models.
 
  - For our tree based models, we needed to encode the time series structure into the dataset for improved forecasting. We opted to include lag data that covered the spectrum of our statistically significant autoregressive terms, namely a t-1 and t-2 lag as well as a t-12 lag to capture any yearly seasonal trends in the data. We ran RandomForestRegression and XGBoost twice for each stock, once with only the stock data and another with the included exogenous variables. We also tuned hyperparameters for each model using gridsearch.
 
  - In an attempt to experiment, we also created a hybrid model utilizing the best SARIMAX parameters and XGBoost. We first modeled the data using SARIMAX, and then modeled the residuals of the data using XGBoost. This mildly improved the performance of our SARIMAX models.
 
  - Our final family of models were LSTMS trained on the stock data, with the same lag parameters as our tree based family. As before, we modeled first only the stock data and then included the data from our exogenous variables. Unsurprisingly, this family of models performed the best and even had relatively good forecasts for the beginning of the target period.
    ![EQT LSTM with Exogenous Variables](https://github.com/jordan-bennett02/Pittsburgh-Stock-Response-Local-Economic-Indicators/blob/main/EQT%20LSTM.png)
 
  - The metrics we utilized to gauge each model was the RMSE and the Max Absolute Error. Beacuse we are modeling percent change in stock values, these metrics aren't the most appropriate metrics to utilize if our models were used for forecasting and profit maximization. You can see in the graph above while the model captures the trend of the data, mathematically it is more important for the model to predict the sign of the percent change. If one wanted to use the above modeling for forecasting for profit maximization, a more viable metric would be a weighted sum of indicator functions that return 1 when the sign on the prediction matches the actual percent change with 0 otherwise. One would want to use the model that maximizes this metric. For our purposes, we seek only to compare the performance of each model with and without the presence of these exogenous variables and so an an average metric seems the most appropriate. We also compute the maximum error for each model.
 
----------------------------------------------------
# Model Performance

| PNC Models | RMSE | Max Error| PPG Models | RMSE | Max Error | EQT Models | RMSE | Max Error |
|------------|------|----------|------------|------|-----------|------------|------|-----------|
|ARIMA (1,0,0)| 6.6788 |10.7829| ARIMA (2,0,0) | 4.7629 | 10.0517 | ARIMA (1,0,0)| 7.6385 | 15.7002 |
|SARIMA (1,0,0), (3,1,0,12)| 6.3673 | 10.6739| SARIMA (2,0,0), (4,1,0,12) | 5.9812 | 12.7273 | SARIMA (1,0,0), (2,1,0,12) | 11.4688 | 19.5740 |
|SARIMAX (1,0,0), (3,1,0,12)|6.3249 | 10.8575| SARIMAX (2,0,0), (4,1,0,12) | 6.0651 | 13.2337 | SARIMAX (1,0,0), (2,1,0,12) | 10.7346 | 19.5638 |
|RF| 6.8906 | 11.7453 | RF | 4.9672 | 11.3359 | RF | 7.9398 | 15.1252 |
|RF with Ex| 6.5191 | 10.0036 | RF with Ex | 4.8565 | 10.4117 | RF with Ex | 9.0174 | 15.8055 | 
|XGBoost | 6.8543 | 11.1830 | XGBoost | 4.8892 | 10.8080 | XGBoost | 7.9255 | 15.2462 |
|XGBoost with Ex | 6.5239 | 9.7682 | XGBoost with Ex | 4.7554 | 9.7492 | XGBoost with Ex | 8.2230 | 15.6964 |
|SARIMAX with XGB Residuals | 6.3010 | 10.5845 | SARIMAX with XGB Residuals | 4.5666 | 9.5830 | SARIMAX with XGB Residuals | 7.7831 | 15.6453 |
|**LSTM** | 5.7627 | **9.5701** | **LSTM** | **4.2794** | 10.0447 | **LSTM** | **6.6195** | 12.8124 |
|**LSTMX** | **5.2856** | 9.7363 | **LSTMX** | 4.3353 | **9.1738** | **LSTMX** | 6.8866 | **12.7954** |

------------------------------------------------------
# Results
- Overall, we find that both of the LSTM models have the best performance metrics with respect to all stocks.
------------------------------------------------------
# Future Improvements
- 
