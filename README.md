# Time Series Analysis of Import/Export Price Indexes: Central/South America

This is a project for Time Series Analysis course at METU. The data that is used for this project can be found [here.](https://data.nasdaq.com/data/BLSN/EIUIH14243-importexport-price-indexes-centralsouth-america-dec-2009100) 

## Steps in the Project:

1.	Time series plot and interpretation (Visually determine the existence of a trend, seasonality, outliers).
2.	Cross-vaidation: Keep several observations out of the analysis to use them to measure the forecast accuracy of the models. 
3.	Box-Cox transformation analysis: If the series need any transformation, do it. If the information criterion values are too close to each other, don’t transform the data.
4.	Make a anomaly detection and if necessary clean the series from anomalies (use anomalize, forecast (tsclean function) or AnomalyDetection packages). 
5.	ACF, PACF plots, KPSS and ADF or PP test results for zero mean, mean and trend cases and their interpretation. For seasonal unit root, HEGY and OCSB or Canova-Hansen tests are required.
6.	If there is a trend, remove it either by detrending or differencing. You may need to apply unit root tests again until observing stationary series.
7.	Then, look at the time series plot of a stationary series, ACF and PACF plots, information table, ESACF (last two are for non-seasonal series).
8.	Identify a proper ARMA or ARIMA model or SARIMA model.
9.	After deciding the order of the possible model (s), run MLE or conditional or uncondinitional LSE and estimate the parameters. Compare the information criteria of several models.
10.	Diagnostic Checking: 
a)	On the residuals, perform portmanteau lack of fit test, look at the ACF-PACF plots of the resuduals, look at the standardized residuals vs time plot to see any outliers or pattern. 
b)	Use histogram, QQ-plot and Shapiro-Wilk test (in ts analysis, economists prefer Jarque-Bera test) to check normality of residuals. 
c)	Perform Breusch-Godfrey test for possible autocorrelation in residual series. The result should be insignificant.
d)	For the Heteroscedasticity, look at the ACF-PACF plots of the squared residuals (there should be no significant spikes); perform ARCH Engle's Test for Residual Heteroscedasticity under aTSA package. The result should be insignificant. If the result is significant, you can state that the error variance is not constant and it should be modelled, but don’t intend to model the variance.
11.	Forecasting: 
a.	Perform Minimum MSE Forecast for the stochastic models.
b.	Use ets code under the forecast package to choose the best exponential smoothing (simple, Holt’s, Holt-Winter’s) method that suits your series for deterministic forecasting. 
c.	Obtain forecasts using Prophet.
d.	Obtain forecasts using TBATS.
e.	Obtain forecasts using neural networks (nnetar)
12.	If you transformed the series for SARIMA model, back transform the series to reach the estimates for the original units. Don’t forget to transform the prediction limits.
13.	Calculate the forecast accuracy measures and state which model gives the highest performance for your dataset.
14.	Provide plots of the original time series, predictions, forecasts and prediction intervals on the same plot drawing the forecast origin for ARIMA models, exponential smoothing method, Prophet, TBATS and neural networks. 


