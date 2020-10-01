import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet

st.title('Forecast 311 Service Calls')

neighborhood_selection = st.selectbox(
    'Which Neighborhood Are We Looking At?',
 ('OHARE', 'ENGLEWOOD', 'WASHINGTON HEIGHTS,ROSELAND',
       'IRVING PARK,AVONDALE', 'SOUTH SHORE, GRAND CROSSING',
       'MARQUETTE PARK,GAGE PARK', 'AUSTIN', 'BRIGHTON PARK,MCKINLEY PARK',
       'MIDWAY AIRPORT', 'HUMBOLDT PARK'))

department_selection = st.selectbox('Which Department Are We Forecasting?',
    ('DOB - Buildings', 'Aviation', 'Streets and Sanitation',
       'DWM - Department of Water Management',
       'CDOT - Department of Transportation'))

days_selection = st.number_input('How many days out are we predicting?', min_value = 1)
#days = int(days)
df = pd.read_pickle('hood_census_sample.pkl')

df.CREATED_DATE = df.CREATED_DATE.apply(lambda x: pd.to_datetime(x))

def create_time_series(neighborhood, department, days):
    df_test = pd.DataFrame(df[(df.NEIGHBORHOOD == neighborhood) & (df.OWNER_DEPARTMENT == department)]
                       .groupby(['CREATED_DATE'])['Unnamed: 0'].count()).reset_index()

    df_test = df_test.set_index('CREATED_DATE').resample('D').first()
    df_test.fillna(0, inplace = True)
    d = {'ds': df_test.index, 'y': df_test['Unnamed: 0']}
    time_series_df = pd.DataFrame(data = d).dropna()
    model = Prophet()
    model.fit(time_series_df)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    f = {'Dates' : forecast.ds, 'Predicted Number of Service Calls' : forecast.yhat*10,
    'Lower Bound': forecast.yhat_lower * 10, 'Upper Bound': forecast.yhat_upper * 10}
    forecast_df = pd.DataFrame(data = f)

    return forecast_df[forecast_df.Dates>'2020-08-31']


time_series = create_time_series(neighborhood_selection, department_selection, days_selection)

st.dataframe(data = time_series)

plot_obj = pd.DataFrame(data = {'Date':time_series.Dates,
        'Forecast' : time_series['Predicted Number of Service Calls']})

plt.plot(plot_obj.Date, plot_obj.Forecast)
plt.xticks(rotation = 45)
plt.title('Forecast: 311 Service Request Volume')
st.pyplot()
#st.line_chart(data=plot_time_series_df.Forecast, width=0, height=0, use_container_width=True)
