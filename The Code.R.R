library(Quandl)
data <- Quandl("BLSN/EIUIH14243", api_key="xrvv_WEYpMEwV4Beoyt2", type= "ts")
data
length(data)

View(data)
frequency(data)
autoplot(data, main = "Plot of Import/Export Price Indexes: Central/South America", xlab= 'Years', y='Import/Export Price Indexes')

train <- window(data, end = c(2019,11))
train
library(forecast)
autoplot(train, main = "Train Data")
test <- window(data, start = c(2019,12))
test
length(test)

l<-BoxCox.lambda(train)
l
train_t<-BoxCox(train,l)
autoplot(train_t, main= "Transformed Data")

#anomaly analysis
library(chron)
time = as.chron(train_t)
time
time1 = as.Date(time,format="%d-%b-%y")
time1 

train_anomaly <- data.frame(trainanomaly = train_t)
head(train_anomaly)

rownames(train_anomaly) = time1
head(train_anomaly)
library(tibble)
library(dplyr)
library(anomalize)
train_tibble <- train_anomaly %>% rownames_to_column() %>% as.tibble() %>% 
  mutate(date = as.Date(rowname)) %>% select(-one_of('rowname'))

head(train_tibble)

train_tibble %>% 
  time_decompose(trainanomaly, method = "stl", frequency = "auto", trend = "auto") %>%
  anomalize(remainder, method = "gesd", alpha = 0.05, max_anoms = 0.2) %>%
  plot_anomaly_decomposition()

train_cleaned <- tsclean(train_t)
train_cleaned
autoplot(train_cleaned, main= "Cleaned and Transformed Data")

ndiffs(train_cleaned) 
nsdiffs(diff(train_cleaned))

#acf, pacf 

ggAcf(train_cleaned, main = "ACF Plot of Train Data ", lag.max = 48)
ggPacf(train_cleaned, main = "PACF Plot of Train Data ", lag.max = 48  ) 

#HEGY Test:
library(uroot)
library(pdR)
mean(train_cleaned) #not zero
mean(diff_train)

hegy.out<-HEGY.test(wts=train_cleaned, itsd=c(1,0,c(1:11)))
hegy.out$stats

hegy.out<-HEGY.test(wts=diff_train, itsd=c(0,0,0))
hegy.out$stats

#Canova-Hansen Test

ch.test(diff_train,type = "dummy",sid=c(1:12))  

# we have monthly data, p is 0.4825, fail to reject the data is stat.

diff_train <- diff(train_cleaned)
diff_train
autoplot(diff_train, main = "Differenced Plot of Import/Export Price Indexes: Central/South America", xlab = 'Years', ylab = 'Import/Export Price Indexes') 

ggAcf(diff_train, main = "ACF Plot of Differenced Data ", lag.max = 48)
ggPacf(diff_train, main = "PACF Plot of Differenced Data ", lag.max=48) 

#seasonal differencing

seasonal_train <- diff(diff_train,12)
autoplot(seasonal_train)
ggAcf(seasonal_train, lag.max = 48, main="ACF Of Seasonal Differencing")
ggPacf(seasonal_train, lag.max = 48, main ="PACF Of Seasonal Differencing")
#sarima(1,1,1)(1,1,1)[12]

fit1<- auto.arima(train_cleaned)
fit1

fit2<-Arima(train_cleaned,order = c(1, 1, 0), seasonal= c(2,0,0))
fit2

fit5<-Arima(train_cleaned,order = c(1, 1, 0), seasonal= c(1,0,0))
fit5

#diagnosis checking

residual <- resid(fit1, standardize = TRUE)
residual
library(ggplot2)

ggAcf(residual,main = "ACF Plot of Residuals ", lag.max = 48)
ggPacf(residual,main = "PACF Plot of Residuals ", lag.max = 48)

autoplot(residual, main='Standardized Residuals')+geom_line(y=0)+theme_minimal()

autoplot(residual)+geom_line(y=0)+theme_minimal()+ggtitle("Plot of The Residuals")+theme_minimal()


#QQ Plot
ggplot(residual, aes(sample = residual)) +stat_qq()+geom_qq_line()+ggtitle("QQ Plot of the Standard Residuals")+theme_minimal()

#histogram
ggplot(residual,aes(x=residual))+geom_histogram(bins=20)+geom_density()+ggtitle("Histogram of the Standard Residuals")+theme_minimal()

#box-plot
summary(residual) #mean and median are not close to each other

ggplot(residual,aes(y=residual,x=as.factor(1)))+geom_boxplot()+ggtitle("Box Plot of the Standard Residuals")+theme_minimal()

#NORMALITY TEST:
#jarque bera test:
jarque.bera.test(residual)
library(tseries)

#shapiro test:
shapiro.test(residual)

#AUTOCORRELATION
#Ljung-Box test:
Box.test(residual,lag=15,type = c("Ljung-Box"))

#Heteroscedasticity
residual_square <- residual ^2
residual_square_acf <-ggAcf(as.vector(residual_square))+theme_minimal()+ggtitle("ACF of Squared Residuals")
residual_square_acf
residual_square_pacf <-ggPacf(as.vector(residual_square))+theme_minimal()+ggtitle("PACF of Squared Residuals")  
residual_square_pacf

#Engle?s ARCH Test:
library(MTS)

archTest(residual)

#FORECASTING:

f<-forecast(fit2,h=12, PI = TRUE)
f

plot(f)
lines(fitted(f), col="blue")
lines(test, col="purple")
abline(v=2020, col="red")
legend("topleft", legend=c("Series", "Fitted Values","Point Forecast","95% Confidence Interval","Test Set"), col=c("Black","Blue", "Light Blue", "Grey", "Purple"), lty=1:2, cex=0.5)

#holt's winter:

fit3 <- ets(train,model="ZZZ") #The algorithm automatically decides type of the components.
fit3 #ETS(M,N,A)

holts_fitted <- forecast(fit3,h=12)
holts_fitted

plot(holts_fitted)
lines(fitted(holts_fitted), col="blue")
lines(test, col="purple")
abline(v=2020, col="red")
legend("topleft", legend=c("Series", "Fitted Values","Point Forecast","95% Confidence Interval","Test Set"), col=c("Black","Blue", "Light Blue", "Grey", "Purple"), lty=1:2, cex=0.5, pch=10, pt.cex = 1)

ets_residual <- resid(holts_fitted)
ets_residual

#jarque-bera
jarque.bera.test(ets_residual)

#shapiro test:
shapiro.test(ets_residual)

#probhet:
library(prophet)
head(train)
ds <- seq(as.Date("2009/12/01"),as.Date("2019/11/01"),by="month")
head(ds)
df<-data.frame(ds,y=as.numeric(train))
head(df)
train_prophet <- prophet(df)
future <- make_future_dataframe(train_prophet,periods = 12)
yhat1 = ts(forecast_prophet["yhat"],start = 2009, frequency = 12)
prediction = forecast_prophet["yhat"]
prediction = ts(prediction, start = 2009, end = 2019, frequency = 12)
tail(future)
dim(df)
dim(future)

forecast_prophet <- predict(train_prophet, future)

tail(forecast_prophet[c('ds', 'yhat', 'yhat_lower', 'yhat_upper')],12)

plot(train_prophet, forecast_prophet)+theme_minimal()+ggtitle("Forecast of Prophet")

accuracy(tail(prediction,12),train)
prophet_test = ts(prediction,start = c(2019,12), end = c(2020,12),frequency = 12)

prophet_residual <- resid(forecast_prophet$yhat)
prophet_residual

#jarque-bera
jarque.bera.test(prophet_residual)

#TBATS:

tbatsmodel<-tbats(train)
tbatsmodel

tbats_forecast<-forecast(tbatsmodel,h=12, PI=TRUE)
tbats_forecast

plot(tbats_forecast)
lines(fitted(tbats_forecast), col="blue")
lines(test, col="purple")
abline(v=2020, col="red")
legend("topleft", legend=c("Series", "Fitted Values","Point Forecast","95% Confidence Interval","Test Set"), col=c("Black","Blue", "Light Blue", "Grey", "Purple"), lty=1:2, cex=0.4)

tbats_residual <- resid(tbats_forecast)
tbats_residual

#jarque-bera
jarque.bera.test(tbats_residual)

#shapiro test:
shapiro.test(tbats_residual)

#neural network:

nnmodel<-nnetar(train)
nnmodel

nnforecast<-forecast(nnmodel,h=12,PI=TRUE)
nnforecast

plot(nnforecast)
lines(fitted(nnforecast), col="blue")
lines(test, col="purple")
abline(v=2020, col="red")
legend("topleft", legend=c("Series", "Fitted Values","Point Forecast","95% Confidence Interval","Test Set"), col=c("Black","Blue", "Light Blue", "Grey", "Purple"), lty=1:2, cex=0.4)


#ACCURACY:

#Since we have SARIMA MODEL, inverse transformation:
f_t<-InvBoxCox(f$mean,l)

accuracy(f,test)
accuracy(f_t,test)
accuracy(holts_fitted,test)
accuracy(tail(prediction,12),train)
accuracy(prophet_test,test)
accuracy(tbats_forecast,test)
accuracy(nnforecast,test)









