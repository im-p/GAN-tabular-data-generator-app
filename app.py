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
from keras.layers import Input as tensor_input
from keras.layers.merge import concatenate
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from sklearn.preprocessing import MinMaxScaler
import flask


server = flask.Flask(__name__)
app = dash.Dash( __name__, server=server, meta_tags=[{"name": "google-site-verification", "content": "WwYO3ZKGp_56A0bc9jiu70QXZ8kubbHcmzvW4VXer9I"}])


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
		multiple=False
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

	#hidden buttons
	html.Div(id = "hidden-buttons", children = [
		#hidden compare data frames button
		html.Button(
			id = "download-button",
			children = html.Div([
				html.A(
					"Download Data",
					id = "download-link",
					download = "generated_data.csv",
					href = "",
					target = "_blank",
				),
			]),
		),
	], className = "one column", style = {"display": "none"}),
])

#about content button
side_panel_about = html.Div([
	html.I("This app uses Generative adversarial network "
		"for generating tabular data. Upload csv or excel file. "
		"Press GAN tab and start GAN training. Make sure data table doesen't contain any NaN values!"
		" You can set few parameters in GAN training: Generator input size"
		" is usually between 1-100. Number of epochs and batch size should be tested for best outcome"),
	html.Br(),
	dcc.Link("More about GANs", href = "https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf"),
	html.Br(),
	dcc.Link("Code", href = "https://github.com/im-p/tabular-data-with-gan"),
	], style = {"text-align": "left"})

#app layout
app.layout = html.Div([

	html.Div([
		html.H2("Tabular data generator with Generative Adversarial Network"),
		], className = "banner"),
	html.Div([
		#side panel
		html.Div([
			dcc.Tabs(id="tabs", value = "About",
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
	#outputs
	html.Div(id = "compare-output"),
	html.Div(id = "compare-output2"),
	html.Div(id = "correlation-output", style = {'display' : 'flex'}),
	#Store dataframes
	dcc.Store(id = "hidden1"),
	dcc.Store(id = "hidden2"),

])


#-----------functions-----------#

#generator
def build_generator(ohe_data, numerical_data_shape, latent_dim):
	outputs = []
	noise = tensor_input(shape = (latent_dim,))
	hidden_1 = Dense(8, kernel_initializer = "he_uniform")(noise)
	hidden_1 = LeakyReLU(0.2)(hidden_1)
	hidden_1 = BatchNormalization(momentum = 0.8)(hidden_1)

	hidden_2 = Dense(16, kernel_initializer = "he_uniform")(hidden_1)
	hidden_2 = LeakyReLU(0.2)(hidden_2)
	hidden_2 = BatchNormalization(momentum = 0.8)(hidden_2)
	#if only numerical data
	if ohe_data is None:
		branch_2 = Dense(32, kernel_initializer = "he_uniform")(hidden_2)
		branch_2 = LeakyReLU(0.2)(branch_2)
		branch_2 = BatchNormalization(momentum=0.8)(branch_2)
		branch_2 = Dense(64, kernel_initializer = "he_uniform")(branch_2)
		branch_2 = LeakyReLU(0.2)(branch_2)
		branch_2 = BatchNormalization(momentum=0.8)(branch_2)
		#Output 1 layer, sigmoid activation for numerical data
		branch_2_output = Dense(numerical_data_shape, activation = "sigmoid")(branch_2)

		return Model(noise, branch_2_output)

	#if numerical and categorical data
	else:
		for data in ohe_data:
			branch_1 = Dense(32, kernel_initializer = "he_uniform")(hidden_2)
			branch_1 = LeakyReLU(0.2)(branch_1)
			branch_1 = BatchNormalization(momentum=0.8)(branch_1)
			#Output 1 layer, softmax activation for multi classification
			branch_1_output = Dense(data.shape[1], activation = "softmax")(branch_1)
			outputs.append(branch_1_output)

		#numerical data
		branch_2 = Dense(32, kernel_initializer = "he_uniform")(hidden_2)
		branch_2 = LeakyReLU(0.2)(branch_2)
		branch_2 = BatchNormalization(momentum=0.8)(branch_2)
		branch_2 = Dense(64, kernel_initializer = "he_uniform")(branch_2)
		branch_2 = LeakyReLU(0.2)(branch_2)
		branch_2 = BatchNormalization(momentum=0.8)(branch_2)
		#Output 1 layer, sigmoid activation for numerical data
		branch_2_output = Dense(numerical_data_shape, activation = "sigmoid")(branch_2)
		outputs.append(branch_2_output)

		combined_output = concatenate([output for output in outputs])

		return Model(noise, combined_output)

#discriminator
def build_discriminator(inputs_n):
	#Input from generator
	d_input = tensor_input(shape = (inputs_n,))
	d = Dense(128, kernel_initializer="he_uniform")(d_input)
	d = LeakyReLU(0.2)(d)
	d = Dense(64, kernel_initializer="he_uniform")(d)
	d = LeakyReLU(0.2)(d)
	d = Dense(32, kernel_initializer="he_uniform")(d)
	d = LeakyReLU(0.2)(d)
	d = Dense(16, kernel_initializer="he_uniform")(d)
	d = LeakyReLU(0.2)(d)
	d = Dense(8, kernel_initializer="he_uniform")(d)
	d = LeakyReLU(0.2)(d)
	#Discriminator output for classification, sigmoid activation
	d_output = Dense(1, activation = "sigmoid")(d)
	#compile and return model
	model = Model(inputs = d_input, outputs = d_output)
	optimizer = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
	return model

#GAN
def build_gan(generator, discriminator):
    #Make discriminator not trainable
    discriminator.trainable = False
    #Discriminator takes input from generator and make discriminator GAN output
    gan_output = discriminator(generator.output)
    #Initialize gan
    model = Model(inputs = generator.input, outputs = gan_output)
    #Compile model
    optimizer = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss = "binary_crossentropy", optimizer = optimizer)
    #Return Model
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
			df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
			return df.to_dict()
		elif "xls" in filename:
			# Assume that the user uploaded an excel file
			df = pd.read_excel(io.BytesIO(decoded))
			return df.to_dict()
	except Exception as e:
		print(e)
		return html.Div([
			"Only csv and excel data is accepted!"
		])

#get categorical data
def get_categorical_data(df):
	categorical_data = []
	ohe_data = []

	#select object data from data frame
	for column in df.select_dtypes("object"):
		categorical_data.append(df[column])
	#append dummies to list
	for column in categorical_data:
		ohe_data.append(pd.get_dummies(column))

	return ohe_data

#slice data 
def get_generated_ohe(ohe_data):
    get_slice = 0
    for index, _ in enumerate(ohe_data):
        get_slice = get_slice + ohe_data[index].shape[1]
    return get_slice

#data table
def data_table(df):
	return html.Div([
		html.H5("Generated data"),
			dash_table.DataTable(
			data = df.to_dict("records"),
			columns=[{"name": i, "id": i} for i in df.columns],
			style_table = {"maxHeight": "400px", "overflowY": "scroll"},
		),
	], className = "bare_container")

#GAN training plot
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

#comparing original and generated data
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
				html.I(f"real data mean {np.round(r[column].mean(), 4)}, real data std {np.round(r[column].std(), 4)}"),
				html.Br(),
				html.I(f"generated data mean {np.round(f[column].mean(), 4)}, generated data std {np.round(f[column].std(), 4)}"),
				dcc.Graph(
					figure = {
						"data": [
							{"x": r_x, "y": r_y, "type": "line", "name": "real distribution"},
							{"x": f_x, "y": f_y, "type": "line", "name": "generated distribution"},
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

#correlation heatmap
def heatmap(og_df, gen_df):
	return html.Div([
		html.Div([
			dcc.Graph(
				figure = {
					"data": [
							{"z": og_df.corr().values, "type": "heatmap"},
					],
					"layout": {
						"title": "Original data correlation matrix"
					}
				}),
			], className = "result_container"),
		html.Div([
			dcc.Graph(
				figure = {
					"data": [
							{"z": gen_df.corr().values, "type": "heatmap"},
					],
					"layout": {
						"title": "Generated data correlation matrix"
					}
				}),
			], className = "result_container")
		], className = "container", style = {'display' : 'flex'})

#original and generated dataframe comparison
def comparison(og_df, gen_df):
	categorical_data = og_df.select_dtypes("object")

	if int(categorical_data.shape[1]) > 0:
		r_df = pd.concat([pd.get_dummies(categorical_data), og_df.select_dtypes("number")], axis = 1)
		return normal_distribution(r_df, gen_df), heatmap(r_df, gen_df)

	else:
		return normal_distribution(og_df, gen_df), heatmap(og_df, gen_df)


#read uploaded file and save data to hidden div
@app.callback(Output("hidden1", "data"),
			[Input("tabs", "value"),
			Input("upload-data", "contents"),
			Input("upload-data", "filename")])
def update_graph(value, contents, filename):
	if value == "GAN" and contents and filename:
		contents = contents
		filename = filename
		df = parse_data(contents, filename)
		return df
	else:
		return None


@app.callback([Output("train-output", "children"),
			Output("gen-data-output", "children"),
			Output("loading-output-1", "children"),
			Output("hidden-buttons", "style"),
			Output("download-link", "style"),
			Output("download-link", "href"),
			Output("compare-output", "children")],
			[Input("hidden1", "data"),
			Input("button1", "n_clicks"),
			Input("latent_dim", "value"),
			Input("n_epochs", "value"),
			Input("n_batch", "value")])

def generate_data(original_data, n_clicks, latent_dim, n_epochs, n_batch):
	"""
	inputs:
		saved original df
		train gan button
		user inputs: latent_dim, n_epochs, n_batch

	outputs:
		plot from training GAN
		generated data table
		displaying hidden elements: download button, download link
		comapareing generated and original dataframes
	"""
	if original_data and n_clicks and latent_dim and n_epochs and n_batch:
		#df from storage
		dff = pd.DataFrame.from_dict(original_data)

		#select numerical data
		numerical_data = dff.select_dtypes("number")

		#rescale numerical data
		mms = MinMaxScaler()
		numerical_data_rescaled = mms.fit_transform(numerical_data)

		#numerical data to df
		numeric = pd.DataFrame(data = numerical_data_rescaled, columns = numerical_data.columns)

		#ohe data
		categorical_data = dff.select_dtypes("object")

		#if categorical and numerical values
		if int(categorical_data.shape[1]) > 0:
			dummies = pd.get_dummies(categorical_data)

			#concatenate data for generator
			data = np.concatenate([dummies.values, numeric.values], axis = 1)

			#columns
			columns = list(dummies.columns) + list(numerical_data.columns)

			#get categorical data shapes for generator
			ohe_data_list = get_categorical_data(dff)

			#build generator
			generator = build_generator(ohe_data_list, numerical_data_rescaled.shape[1], latent_dim)

			#build discriminator
			discriminator = build_discriminator(data.shape[1])

			#build gan
			gan = build_gan(generator, discriminator)
			#train gan
			g_loss, d_loss = train(gan, generator, discriminator, data, latent_dim, n_epochs, n_batch)

			#generate data from the noise
			noise = np.random.normal(0, 1, (data.shape[0], latent_dim))
			generated_data = generator.predict(noise)

			#round softmax values
			get_slice = get_generated_ohe(ohe_data_list)
			generated_ohe_data = np.round(generated_data[:, :get_slice])

			#get numerical data
			generated_numerical_data = generated_data[:, get_slice:]
			
			#inverse transform for numerical data
			generated_numerical_data = mms.inverse_transform(generated_numerical_data)

			#concatenate numpy arrays
			generated_data = np.concatenate([generated_ohe_data, generated_numerical_data], axis = 1)

		#if only numerical data
		else:
			#build generator
			generator = build_generator(None, numeric.shape[1], latent_dim)
			#build discriminator
			discriminator = build_discriminator(numeric.shape[1])
			#build gan
			gan = build_gan(generator, discriminator)
			#train gan
			g_loss, d_loss = train(gan, generator, discriminator, numerical_data_rescaled, latent_dim, n_epochs, n_batch)
			#generate data from the noise
			noise = np.random.normal(0, 1, (numerical_data_rescaled.shape[0], latent_dim))
			generated_data = generator.predict(noise)
			#inverse transform for numerical data
			generated_data = mms.inverse_transform(generated_data)
			columns = numeric.columns


		#make pandas df
		generated_df = pd.DataFrame(data = generated_data, columns = columns)


		#download link
		csv_string = generated_df.to_csv(index = False, encoding = "utf-8")
		csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)

		return plot_training(d_loss, g_loss), data_table(generated_df), html.H5("Generated data is ready!"), {"display": "block"}, {"display": "block"}, csv_string, comparison(dff, generated_df)

	else:
		return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update



if __name__ == "__main__":
	app.run_server(debug=True)