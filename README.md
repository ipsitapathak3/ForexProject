# Foreign Exchange Rate Analysis using Economic Factors with Regression techniques
In a Floating Exchange Rate system, the value of the currency is allowed to fluctuate freely according to changes in demand and supply of foreign exchange, which are determined by several economic factors. The aim of this project is to create a model that predicts USD to INR exchange rates, taking into account the various economic factors that affect the exchange rate.
The two regression analyses are performed over a dataset showing daily USD to INR exchange rates from January 4, 1973 to January 8, 2018. 


LASSO Regression

The model was trained on the data for 35 years, taking the value of λ as 1. Then, it was used to predict exchange rates for the remaining 10 years. The following are the results.
This model has an R-squared score of 0.967 and mean-square error of 2.907. 

Ridge Regression

The model was trained on the data for 35 years, taking the value of λ as 1. Then, it was used to predict exchange rates for the remaining 10 years. The following are the results.
This model has an R-squared score of 0.999 and mean-square error of 0.0821.

Autoregressive Integrated Moving Averages

It can be noted that the model can predict the increasing trend of the time series. The model can also predict the USD to INR exchange rate for the next 8-10 days with higher accuracy.

Conclusion

Overall, all three algorithms perform well to predict the USD to INR exchange rates. 
The LASSO and Ridge Regression models are able to capture the fluctuations in exchange rate (or the trend) much better than the ARIMA model. The ARIMA model can only capture the overall increasing trend. 
However, the ARIMA model can predict the USD to INR exchange rate for the next 10 days with higher accuracy. Further, it does not involve consideration of the previous days’ exchange rates, which are a requirement for the LASSO and Ridge Regression models. To predict future values for which previous days’ exchange rates are unknown, the ARIMA model is a better choice. 
