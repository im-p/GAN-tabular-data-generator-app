import base64
import io

import urllib.parse

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_table

import pandas as pd
import numpy as np
import scipy.stats

from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from sklearn.preprocessing import MinMaxScaler
import flask

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)


#side panel training gan and download generated data
side_panel_gan = html.Div([
	#upload button
	dcc.Upload(
		id = "upload-data",
		children = html.Div([
			"Drag and Drop or ",
			html.A("Select Files")
		]),
		style = {"borderStyle": "dashed", 
			"height": "60px",
			"lineHeight": "60px",
			"borderWidth": "1px",
			"borderStyle": "dashed",
			"borderRadius": "5px",
			"margin-top": "5px"
		},
		# Allow multiple files to be uploaded
		multiple=True
	),
	#training parameters inputs
	html.I("Generators input size:"),
	html.Br(),
	dcc.Input(id = "latent_dim", type = "number", placeholder = "Generator input size", style = {"width": "80%"}),
	html.Br(),
	html.I("Number of epochs:"),
	dcc.Input(id = "n_epochs", type = "number", placeholder = "number of epochs", style = {"width": "80%"}),
	html.Br(),
	html.I("Batch size:"),
	dcc.Input(id = "n_batch", type = "number", placeholder = "batch size", style = {"width": "80%"}),
	html.Button("Train GAN", id = "button1", style = {"margin": "10px"}),
	html.Hr(),
	dcc.Loading(id = "loading-1", children = [html.Div(id = "loading-output-1")]),

	#hidden download button
	html.Div([
		html.Button(
			id = "download-button",
			children = html.Div([
				html.A(
					"Download Data",
					id = "download-link",
					download = "generated_data.csv",
					href = "",
					target = "_blank",
					style = {"display": "none"}
				),
			]),
			style = {"display": "none"},
		),
	], className = "one column", style = {"margin-left": "65px"}),
])

side_panel_about = html.Div([
	html.I("Generative adversarial network "
		"for generating numerical data. "
		"Press GAN tab and start GAN "
		"training. "),
	html.Br(),
	dcc.Link("More about GANs", href = "https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf"),
	html.Br(),
	dcc.Link("Code", href = "https://github.com/im-p/tabular-data-with-gan"),
	], style = {"text-align": "left"})

app.layout = html.Div([

	html.Div([
		html.H2("Numerical data generator"),
		], className = "banner"),
	html.Div([
		#side panel
		html.Div([
			dcc.Tabs(id="tabs", value = "GAN",
					children=[
				        dcc.Tab(label = "About", value = "About", children = side_panel_about),
				        dcc.Tab(label = "GAN", value = "GAN", children = side_panel_gan),
    				]),
			
			], className = "side_bar"),
		#outputs
		html.Div([
			html.Div(id = "train-output"),
			html.Div(id = "gen-data-output"),
		], className = "nine columns"),

	], className = "row", style = {'display' : 'flex'}),

	html.Div(id = "compare-output"),

	html.Div(id = "hidden1", style = {"display": "none"}),
	html.Div(id = "hidden2", style = {"display": "none"}),

])


#-----------functions-----------#

#generator
def build_generator(n_columns, latent_dim):
	model = Sequential()
	model.add(Dense(32, kernel_initializer = "he_uniform", input_dim=latent_dim))
	model.add(LeakyReLU(0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Dense(64,  kernel_initializer = "he_uniform"))
	model.add(LeakyReLU(0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Dense(128,  kernel_initializer = "he_uniform"))
	model.add(LeakyReLU(0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Dense(n_columns, activation = "sigmoid"))
	return model

#discriminator
def build_discriminator(inputs_n):
	model = Sequential()
	model.add(Dense(128,  kernel_initializer = "he_uniform", input_dim = inputs_n))
	model.add(LeakyReLU(0.2))
	model.add(Dense(64,  kernel_initializer = "he_uniform"))
	model.add(LeakyReLU(0.2))
	model.add(Dense(32,  kernel_initializer = "he_uniform"))
	model.add(LeakyReLU(0.2))
	model.add(Dense(16,  kernel_initializer = "he_uniform"))
	model.add(LeakyReLU(0.2))
	model.add(Dense(1, activation = "sigmoid"))
	optimizer = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
	return model

#GAN
def build_gan(generator, discriminator):
	# make weights in the discriminator not trainable
	discriminator.trainable = False
	# connect generator and dicriminator
	model = Sequential()
	# add generator
	model.add(generator)
	# add the discriminator
	model.add(discriminator)
	# compile model
	optimizer = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss = "binary_crossentropy", optimizer = optimizer)
	return model

#GAN training
def train(gan, generator, discriminator, data, latent_dim, n_epochs, n_batch):
	#Half batch size for updateting discriminator
	half_batch = int(n_batch / 2)

	#lists for stats from the model
	generator_loss = []
	discriminator_loss = []

	#generate class labels for fake = 0 and real = 1
	valid = np.ones((half_batch, 1))
	fake = np.zeros((half_batch, 1))
	y_gan = np.ones((n_batch, 1))
	#training loop
	for i in range(n_epochs):
		
		#select random batch from the real numerical data
		idx = np.random.randint(0, data.shape[0], half_batch)
		real_data = data[idx]

		#generate fake samples from the noise
		noise = np.random.normal(0, 1, (half_batch, latent_dim))
		fake_data = generator.predict(noise)

		#train the discriminator and return losses
		d_loss_real, _ = discriminator.train_on_batch(real_data, valid)
		d_loss_fake, _ = discriminator.train_on_batch(fake_data, fake)

		d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
		discriminator_loss.append(d_loss)

		#generate noise for generator input and  train the generator (to have the discriminator label samples as valid)
		noise = np.random.normal(0, 1, (n_batch, latent_dim))
		g_loss = gan.train_on_batch(noise, y_gan)
		generator_loss.append(g_loss)

	return generator_loss, discriminator_loss

#read uploaded file
def parse_data(contents, filename):
	content_type, content_string = contents.split(",")

	decoded = base64.b64decode(content_string)
	try:
		if "csv" in filename:
			# Assume that the user uploaded a CSV or TXT file
			df = pd.read_csv(
				io.StringIO(decoded.decode("utf-8")))
		elif "xls" in filename:
			# Assume that the user uploaded an excel file
			df = pd.read_excel(io.BytesIO(decoded))

	except Exception as e:
		print(e)
		return html.Div([
			"Only csv and excel data is accepted!"
		])

	return df.to_dict()

#select numerical data, preprocess data, train gan, generate new data
def train_gan(df, latent_dim, n_epochs, n_batch):
	#select numerical data
	df = df.select_dtypes("number")
			
	#get column names
	columns = df.columns
	#rescale data
	mms = MinMaxScaler()
	df = mms.fit_transform(df)
	#build generator
	generator = build_generator(df.shape[1], latent_dim)
	#build discriminator
	discriminator = build_discriminator(df.shape[1])
	#build gan
	gan = build_gan(generator, discriminator)
	#train gan
	g_loss, d_loss = train(gan, generator, discriminator, df, latent_dim, n_epochs, n_batch)
	#generate data
	noise = np.random.normal(0, 1, (df.shape[0], latent_dim))
	generated_data = generator.predict(noise)
	#data to original form
	generated_data = mms.inverse_transform(generated_data)
	generated_df = pd.DataFrame(data = generated_data, columns = columns)

	return generated_df, g_loss, d_loss

def data_table(df):
	return html.Div([
		html.H5("Generated data"),
			dash_table.DataTable(
			data = df.to_dict("records"),
			columns=[{"name": i, "id": i} for i in df.columns],
			style_table = {"maxHeight": "400px", "overflowY": "scroll"},
		),
	], className = "bare_container")

def plot_training(d_loss, g_loss):
	return html.Div([
			dcc.Graph(
				figure = {
					"data": [
						{"y": d_loss, "type": "line", "name": "discriminator loss"},
						{"y": g_loss, "type": "line", "name": "generator loss"}
					],
					"layout": {
						"title": "GAN training progression",
						"borderRadius": "5px",
						"legend": dict(orientation = "h"),
						"height": "300"
						}
				}),
			], className = "bare_container"),

def normal_distribution(r, f):
	plots = []
	hist = []
	for column in r.columns:
		r_x = np.linspace(r[column].min(), r[column].max(), len(r[column]))
		f_x = np.linspace(f[column].min(), f[column].max(), len(f[column]))

		r_y = scipy.stats.norm.pdf(r_x, r[column].mean(), r[column].std())
		f_y = scipy.stats.norm.pdf(f_x, f[column].mean(), f[column].std())

		plots.append(
			html.Div([
				html.I(f"Original data mean {np.round(r[column].mean(), 4)}, Original data std {np.round(r[column].std(), 4)}"),
				html.I(f"Generated data mean {np.round(f[column].mean(), 4)}, Generated data std {np.round(f[column].std(), 4)}"),
				dcc.Graph(
					figure = {
						"data": [
							{"x": r_x, "y": r_y, "type": "line", "name": "real distribution"},
							{"x": f_x, "y": f_y, "type": "line", "name": "fake distribution"},
						],
						"layout": {
							"title": column + " normal distribution",
							"borderRadius": "5px",
							"legend": dict(orientation = "h"),
							"height": "300"
							}
					}),
				])
		)

		hist.append(
			html.Div([
				dcc.Graph(
					figure = {
						"data": [
							{"x": r[column], "type": "histogram", "name": "real data"},
							{"x": f[column], "type": "histogram", "name": "generated data"},
						],
						"layout": {
							"title": column + " data distribution",
							"borderRadius": "5px",
							"legend": dict(orientation = "h"),
							"height": "300"
							}
					}),
				html.Hr(),
				]),
		)

	return html.Div([html.Div(plots, className = "result_container"), html.Div(hist, className = "result_container")], className = "row", style = {'display' : 'flex'})


#read uploaded file and save data to hidden div
@app.callback(Output("hidden1", "children"),
			[Input("tabs", "value"),
			Input("upload-data", "contents"),
			Input("upload-data", "filename")])
def update_graph(value, contents, filename):
	if value == "GAN" and contents and filename:
		contents = contents[0]
		filename = filename[0]
		df = parse_data(contents, filename)
		return df
	else:
		return None

#train gan and get new generated data. return datatable, losses, download link
@app.callback([Output("train-output", "children"),
			Output("gen-data-output", "children"),
			Output("loading-output-1", "children"),
			Output("hidden2", "children"),
			Output("download-button", component_property = "style"),
			Output("download-link", component_property = "style"),
			Output("download-link", "href"),
			Output("compare-output", "children")],
			[Input("hidden1", "children"),
			Input("button1", "n_clicks"),
			Input("latent_dim", "value"),
			Input("n_epochs", "value"),
			Input("n_batch", "value")])

def generate_data(original_data, n_clicks, latent_dim, n_epochs, n_batch):
	if original_data and n_clicks and latent_dim and n_epochs and n_batch:
		#df from hidden div
		dff = pd.DataFrame.from_dict(original_data)
		#use gan to generate new data
		generated_df, g_loss, d_loss = train_gan(dff, latent_dim, n_epochs, n_batch)

		#download link
		csv_string = generated_df.to_csv(index=False, encoding="utf-8")
		csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)

		return plot_training(d_loss, g_loss), data_table(generated_df), html.H5("Generated data is ready!"), generated_df.to_dict(), {"display": "block"}, {"display": "block"}, csv_string, normal_distribution(dff, generated_df)

	else:
		return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

#reset inputs
@app.callback([Output("latent_dim", "value"),
			Output("n_epochs", "value"),
			Output("n_batch", "value"),
			Output("button1", "n_clicks")],
			[Input("hidden2", "children")])

def clear_inputs(generated_data):
	if generated_data:
		return None, None, None, None
	else:
		return dash.no_update, dash.no_update, dash.no_update, dash.no_update

if __name__ == "__main__":
	app.run_server(debug=True)