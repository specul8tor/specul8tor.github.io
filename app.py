import pandas_datareader.data as web
import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
from dash.dependencies import Input, Output, State
from main import DataPrep, ModelDesign, Prediction, Forecast
import pandas as pd

def StockData(start,end,input_data,value):
	start = datetime.datetime.strptime(start, "%Y-%m-%d")
	end =datetime.datetime.strptime(end, "%Y-%m-%d")

	df = web.DataReader(input_data, 'yahoo',start, end)

	if (value==1):
		data = df['Open']
	elif (value==2):
		data = (df['High']+df['Low'])/2
	elif (value==3):
		data = df['Adj Close']

	start=start.strftime("%Y-%m-%d")
	end=end.strftime("%Y-%m-%d")
	dff = df.reset_index()
	time = dff['Date']
	date=[]

	for i in range(len(time)):
		t=pd.to_datetime(str(time[i]))
		date.append(t.strftime('%Y-%m-%d'))

	return df,dff,data,date,start,end

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container([

	dbc.Row(
		dbc.Col([
			html.H1(
			)
		],xl={'size':12})
	),

	dbc.Row(
		dbc.Col([
			html.H1(
			)
		],xl={'size':12})
	),

	dbc.Row(        
		dbc.Col(
			html.H1(
				"Stock Market Dashboard",
				className='text-center text-primary mb-4'
			),xl=12)
	),

	dbc.Row(
		dbc.Col([
			html.H1(
			)
		],xl={'size':12})
	),

	dcc.Tabs(colors={ 'border': '#d6d6d6', 'primary': '#000000', 'background': '#ffffff', },children=[

		dcc.Tab(label='Inputs', children=[

			dbc.Row(
				dbc.Col([
					html.H1(
					)
				],xl={'size':12})
			),

			dbc.Row(
				dbc.Col([
					html.H1(
					)
				],xl={'size':12})
			),

			dbc.Row([

				dbc.Col([

					dcc.Input(
						id='LSTM-layers', 
						value='', 
						type='text',
						placeholder='# of LSTM Layers',
						style={'height':'40px', 'width':'304px'},
						disabled=True
				    ),
					dcc.Input(
						id='LSTM-layers-input', 
						value='', 
						type='text',
						style={'height':'40px', 'width':'85px'},
				    ),  
					dcc.Input(
						id='neurons', 
						value='', 
						type='text',
						placeholder='# of Neurons in each LSTM Layer',
						style={'height':'40px', 'width':'304px'},
						disabled=True
					),
				    dcc.Input(
				    	id='neurons-input', 
				    	value='', 
				    	type='text',
				    	style={'height':'40px', 'width':'85px'},
				    ),			    
					dcc.Input(
						id='Dropout-layers', 
						value='', 
						type='text',
						placeholder='# of Dropout Layers',
						style={'height':'40px', 'width':'304px'},
						disabled=True
				    ), 
				    dcc.Input(
				    	id='Dropout-layers-input', 
				    	value='', 
				    	type='text',
				    	style={'height':'40px', 'width':'85px'},
				    ),
					dcc.Input(
						id='Dropout-number', 
						value='', 
						type='text',
						placeholder='Amount of Dropout (in decimal)',
						style={'height':'40px', 'width':'304px'},
						disabled=True
				    ),
				    dcc.Input(
				    	id='Dropout-number-input', 
				    	value='', 
				    	type='text',
				    	style={'height':'40px', 'width':'85px'},
				    ),
					dcc.Input(
						id='batch-size', 
						value='', 
						type='text',
						placeholder='Batch Size',
						style={'height':'40px', 'width':'304px'},
						disabled=True
				    ), 
				    dcc.Input(
				    	id='batch-size-input', 
				    	value='', 
				    	type='text',
				    	style={'height':'40px', 'width':'85px'},
				    ),
					dcc.Input(
						id='epochs', 
						value='', 
						type='text',
						placeholder='# of Epochs',
						style={'height':'40px', 'width':'304px'},
						disabled=True
				    ),
				    dcc.Input(
				    	id='epochs-input', 
				    	value='', 
				    	type='text',
				    	style={'height':'40px', 'width':'85px'},
				    ),
					dcc.Input(
						id='days', 
						value='', 
						type='text',
						placeholder='# of Previous Days to Use',
						style={'height':'40px', 'width':'304px'},
						disabled=True
				    ),
				    dcc.Input(
				    	id='days-input', 
				    	value='', 
				    	type='text',
				    	style={'height':'40px', 'width':'85px'},
				    ),
				    dcc.Input(
				    	id='future', 
				    	value='', 
				    	type='text',
				    	placeholder='# of Days to Forecast',
				    	style={'height':'40px', 'width':'304px'},
				    	disabled=True
				    ),
				    dcc.Input(
				    	id='future-input', 
				    	value='', 
				    	type='text',
				    	style={'height':'40px', 'width':'85px'},
				    ),
				    dcc.Dropdown(
						id='activation-dropdown',
						options=[
						    {'label': 'Relu', 'value': 'relu'},
						    {'label': 'Sigmoid', 'value': 'sigmoid'},
						    {'label': 'Tanh', 'value': 'tanh'}
						],
						placeholder='Type of Activation',
						style={'width':'390px'}
					),
				    dcc.Dropdown(
						id='optimizer-dropdown',
						options=[
						    {'label': 'Adam', 'value': 'adam'},
						    {'label': 'Stochastic Gradient Descent', 'value': 'sgd'},
						    {'label': 'RMSprop', 'value': 'RMSprop'}
						],
						placeholder='Type of Optimizer',
						style={'width':'390px'}
					),
					dcc.Dropdown(
						id='loss-dropdown',
						options=[
						    {'label': 'Mean Squared Error', 'value': 'mse'},
						    {'label': 'Catagorical Cross Entropy', 'value': 'categorical_crossentropy'},
						    {'label': 'Mean Absolute Error', 'value': 'mae'}
						],
						placeholder='Type of Loss Function',
						style={'width':'390px'}
						),

					dcc.Dropdown(
						id='price-dropdown',
						options=[
						    {'label': 'Open', 'value': 1},
						    {'label': 'High/Low Average', 'value': 2},
						    {'label': 'Adjusted Close', 'value': 3}
						],
						placeholder='Type of Price',
						style={'width':'390px'}
					),
					html.H1(
						),
					html.H1(
						),
					dcc.Slider(
						id='slider',
						min=0,
						max=1,
						step=0.01,
						tooltip={'always_visible': True},
						value=0.8
					)	
				],xl={'offset':1, 'size':4}),

				dbc.Col([
				    dcc.Input(
				    	id='ticker1', 
				    	value='', 
				    	type='text',
				    	placeholder='Ticker 1',
				    	style={'height':'40px', 'width':'100px'},
				    	disabled=True
				    ),
					dcc.Input(
						id='stock-input1', 
						value='',
						placeholder='Stock Tickers', 
						type='text',
						style={'width':'92px', 'height': '40px'}
					),
					dcc.DatePickerRange(
						id='date1',
					),

				    dcc.Input(
				    	id='ticker2', 
				    	value='', 
				    	type='text',
				    	placeholder='Ticker 2',
				    	style={'height':'40px', 'width':'100px'},
				    	disabled=True
				    ),
					dcc.Input(
						id='stock-input2', 
						value='',
						placeholder='Stock Tickers', 
						type='text',
						style={'width':'92px', 'height': '40px'}
					),
					dcc.DatePickerRange(
						id='date2',
					),

				    dcc.Input(
				    	id='ticker3', 
				    	value='', 
				    	type='text',
				    	placeholder='Ticker 3',
				    	style={'height':'40px', 'width':'100px'},
				    	disabled=True
				    ),
					dcc.Input(
						id='stock-input3', 
						value='',
						placeholder='Stock Tickers', 
						type='text',
						style={'width':'92px', 'height': '40px'}
					),
					dcc.DatePickerRange(
						id='date3',
					),

				    dcc.Input(
				    	id='ticker4', 
				    	value='', 
				    	type='text',
				    	placeholder='Ticker 4',
				    	style={'height':'40px', 'width':'100px'},
				    	disabled=True
				    ),
					dcc.Input(
						id='stock-input4', 
						value='',
						placeholder='Stock Tickers', 
						type='text',
						style={'width':'92px', 'height': '40px'}
					),
					dcc.DatePickerRange(
						id='date4',
					),

				    dcc.Input(
				    	id='ticker5', 
				    	value='', 
				    	type='text',
				    	placeholder='Ticker 5',
				    	style={'height':'40px', 'width':'100px'},
				    	disabled=True
				    ),
					dcc.Input(
						id='stock-input5', 
						value='',
						placeholder='Stock Tickers', 
						type='text',
						style={'width':'92px', 'height': '40px'}
					),
					dcc.DatePickerRange(
						id='date5',
					),
					html.Div(dbc.Button("Run", id='button', className="mr-2", color='primary', style={'width':'478px'}))

				],xl={'offset':2, 'size':5}),
			]),

			dbc.Row(
				dbc.Col([
					html.H1(
					)
				],xl={'size':12})
			),

			]),
		dcc.Tab(label='Outputs', children=[

			dbc.Row(
				dbc.Col([
					html.H1(
					)
				],xl={'size':12})
			),

			dbc.Row(
				dbc.Col([
					html.H1(
					)
				],xl={'size':12})
			),

			dbc.Spinner(children=[html.Div(id="output")], size="lg", color="primary", type="border", fullscreen=True,),

			])
	])
], fluid=True)

@app.callback(

    Output(component_id='output', component_property='children')
    ,
    [Input(component_id='button', component_property='n_clicks')
	,State(component_id='LSTM-layers-input', component_property='value')
	,State(component_id='neurons-input', component_property='value')
	,State(component_id='Dropout-layers-input', component_property='value')
	,State(component_id='Dropout-number-input', component_property='value')
	,State(component_id='batch-size-input', component_property='value')
	,State(component_id='epochs-input', component_property='value')
	,State(component_id='days-input', component_property='value')
	,State(component_id='future-input',component_property='value')
	,State(component_id='activation-dropdown', component_property='value')
	,State(component_id='optimizer-dropdown', component_property='value')
	,State(component_id='loss-dropdown', component_property='value')
	,State(component_id='price-dropdown',component_property='value')
	,State(component_id='slider', component_property='value')
	,State(component_id='stock-input1', component_property='value')
	,State(component_id='date1', component_property='start_date')
	,State(component_id='date1', component_property='end_date')
	,State(component_id='stock-input2', component_property='value')
	,State(component_id='date2', component_property='start_date')
	,State(component_id='date2', component_property='end_date')
	,State(component_id='stock-input3', component_property='value')
	,State(component_id='date3', component_property='start_date')
	,State(component_id='date3', component_property='end_date')
	,State(component_id='stock-input4', component_property='value')
	,State(component_id='date4', component_property='start_date')
	,State(component_id='date4', component_property='end_date')
	,State(component_id='stock-input5', component_property='value')
	,State(component_id='date5', component_property='start_date')
	,State(component_id='date5', component_property='end_date')
    ]
)

def update_graph(n_clicks,LSTM_layers,neurons,Dropout_layers,Dropout_number,batch_size,epochs,days,future,activation,optimizer,loss,value,slider,input_data,start1,end1,input2,start2,end2,input3,start3,end3,input4,start4,end4,input5,start5,end5):

	input_array = [input_data,input2,input3,input4,input5]
	start = [start1,start2,start3,start4,start5]
	end = [end1,end2,end3,end4,end5]

	for z in list(input_array):
		if z == '':
			input_array.remove(z)
			start.remove(None)
			end.remove(None)

	send=[]
	for i in range(len(input_array)):
		df,dff,data,date,_,_ = StockData(start[i],end[i],input_array[i],value)

		start_train = 0
		end_train =int((len(date)-1)*float(slider))

		data = np.reshape(data,(data.shape[0]))
		data = data.tolist()

		data,x_train,y_train,x_test,y_test,maximum,start_train,end_train,data_list = DataPrep(data,int(days),start_train,end_train,1)
		trained_model,model=ModelDesign(int(LSTM_layers),int(Dropout_layers),float(Dropout_number),int(neurons),int(batch_size),int(epochs),x_train,y_train,activation,optimizer,loss)
		test_predictions, train_preditions = Prediction(x_test,x_train,model,1)
		data=data*maximum
		data = data.tolist()

		if(len(date[end_train:])-1 !=0):
			test_predictions=np.reshape(test_predictions,(test_predictions.shape[0]))
			test_predictions=test_predictions*maximum
			test_predictions=test_predictions.tolist()
			y_test = y_test*maximum
			y_test = y_test.tolist()
		else:
			test_predictions = []
			y_test = []

		train_preditions=train_preditions*maximum
		train_preditions=np.reshape(train_preditions,(train_preditions.shape[0]))
		train_preditions=train_preditions.tolist()

		lossList=[]
		for j in trained_model.history['loss']:
			lossList.append(j)

		if(int(future) != 0):
			future_predictions, future_data = Forecast(int(future),model,data_list,int(days))
			future_predictions = future_predictions*maximum
			future_predictions=future_predictions.tolist()
		else:
			future_predictions=[]

		output_main=html.Div(children=[
			dbc.Row([
				dbc.Col([
					dcc.Graph(
						id = 'example-graph',
						figure = {
						    'data': [
							    {'x': date, 'y': data, 'type': 'line', 'name': 'Actual'}
							    ,{'x': date[start_train:end_train], 'y': train_preditions, 'type': 'line', 'name': 'Training'}
							    ,{'x': date[end_train+1:], 'y': test_predictions, 'type': 'line', 'name': 'Test'}
						    ],
						    'layout': {'title': input_array[i],'xaxis':{'title':'Date'},'yaxis':{'title':'Price'}}
						}
					)
				],xl={'size':12, 'offset':0})
			]),
			dbc.Row([

				dbc.Col([
					dcc.Graph(
						id = 'ex-loss-graph',
						figure = {
						    'data': [
						    {'x': list(range(int(epochs))), 'y': lossList, 'type': 'line', 'name': 'Actual'}
						    ],
						    'layout': {'title': 'Loss','xaxis':{'title':'Epochs'},'yaxis':{'title':'Loss'}}
						}
					)

				],xl={'size':6, 'offset':0}),

				dbc.Col([
					dcc.Graph(
						id = 'ex-future-graph',
						figure = {
						    'data': [
							{'x':list(range(int(future))),'y': data[len(data)-int(future):], 'type': 'line', 'name': 'Actual'}
							,{'x':list(range(int(future))),'y':test_predictions[len(test_predictions)-int(future):],'type':'line','name':'Test','line':dict(color='green')}
						    ,{ 'x':list(range(int(future),int(future)*2)),'y': future_predictions, 'type': 'line', 'name': 'Forecast','line':dict(color='red')}

						    ],
						    'layout': {'title': 'Future Forecast','xaxis':{'title':'Days'},'yaxis':{'title':'Price'}}
						}
					)

				],xl={'size':6, 'offset':0})					

			])
		])

		send.append(output_main)

	return send

if __name__ == '__main__':
    app.run_server(debug=True)