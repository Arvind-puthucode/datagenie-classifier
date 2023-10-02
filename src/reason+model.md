# Characteristics of Time Series Models

## Trend and Seasonality

- **LSTM:**
  - Can capture complex, non-linear trends and seasonal patterns.
  
- **Prophet:**
  - Designed to model time series with strong seasonal patterns and multiple seasonalities.
  
- **ARIMA:**
  - Handles linear trends and seasonal patterns.
  
- **Exponential Smoothing (exps):**
  - Suitable for data with a constant or changing trend, and can capture some seasonal patterns.

## Noise Level

- **LSTM:**
  - Can handle data with higher noise levels due to its ability to model complex relationships.
  
- **ARIMA:**
  - Effective for data with moderate noise levels.
  
- **Exponential Smoothing (exps):**
  - Works well for data with lower noise levels.

## Temporal Dependencies

- **LSTM:**
  - Can capture long-term temporal dependencies.
  
- **ARIMA:**
  - Models temporal dependencies using autoregressive and moving average terms.
  
- **Prophet:**
  - Accounts for changes in trends and seasonalities over time.
  
- **Exponential Smoothing (exps):**
  - Models temporal dependencies based on previous observations.

## Data Stationarity

- **LSTM:**
  - Doesn't require data to be stationary but may benefit from preprocessing.
  
- **ARIMA:**
  - Requires the data to be stationary or made stationary through differencing.
  
- **Exponential Smoothing (exps):**
  - Typically works well with non-stationary data.

## Model Complexity

- **LSTM:**
  - Complex model with a large number of parameters, suitable for complex data patterns.
  
- **ARIMA:**
  - Relatively simpler model compared to LSTM but still effective for many time series patterns.
  
- **Prophet:**
  - Provides a balance between simplicity and effectiveness for time series with strong seasonalities and holiday effects.
  
- **Exponential Smoothing (exps):**
  - Simple and efficient, especially for forecasting with smoothing parameters.

## Data Size and Availability

- For smaller datasets or when computational resources are limited, simpler models like ARIMA or exponential smoothing may be preferred.

## Historical Data Behavior

- Understanding the historical data behavior can help in selecting an appropriate model.

