from flask import Flask ,render_template,session,url_for,redirect
import numpy as np
from flask_wtf import FlaskForm
from wtforms import TextField, SubmitField
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.losses import sparse_categorical_crossentropy #for one-hot-encoded features
#import joblib

def generate_text(model, input_json):
  char_to_ind = {'\n': 0, ' ': 1, '!': 2, '"': 3, '&': 4, "'": 5, '(': 6, ')': 7,
                ',': 8, '-': 9, '.': 10, '0': 11, '1': 12, '2': 13, '3': 14, 
                '4': 15, '5': 16, '6': 17, '7': 18, '8': 19, '9': 20, ':': 21,
                ';': 22, '<': 23, '>': 24, '?': 25, 'A': 26, 'B': 27, 'C': 28,
                'D': 29, 'E': 30, 'F': 31, 'G': 32, 'H': 33, 'I': 34, 'J': 35,
                'K': 36, 'L': 37, 'M': 38, 'N': 39, 'O': 40, 'P': 41, 'Q': 42,
                'R': 43, 'S': 44, 'T': 45, 'U': 46, 'V': 47, 'W': 48, 'X': 49,
                'Y': 50, 'Z': 51, '[': 52, ']': 53, '_': 54, '`': 55, 'a': 56,
                'b': 57, 'c': 58, 'd': 59, 'e': 60, 'f': 61, 'g': 62, 'h': 63,
                'i': 64, 'j': 65, 'k': 66, 'l': 67, 'm': 68, 'n': 69, 'o': 70,
                'p': 71, 'q': 72, 'r': 73, 's': 74, 't': 75, 'u': 76, 'v': 77,
                'w': 78, 'x': 79, 'y': 80, 'z': 81, '|': 82,  '}': 83}

  ind_to_char = ['\n', ' ', '!', '"', '&', "'", '(', ')', ',', '-', '.', '0', '1',
       '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '>', '?',
       'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
       'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
       '[', ']', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
       'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
       'w', 'x', 'y', 'z', '|', '}']

  start_seed = input_json["start_seed"]
  gen_size = input_json["gen_size"]
  temp = input_json["temp"]

  num_generate = gen_size
  input_eval = [char_to_ind[s] for s in start_seed]

  input_eval = tf.expand_dims(input_eval, 0)

  text_generated=[]

  temperature = temp

  model.reset_states()

  for i in range(num_generate):
    predictions = model(input_eval)

    predictions = tf.squeeze(predictions, 0)

    predictions = predictions/temperature

    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

    input_eval = tf.expand_dims([predicted_id], 0)

    text_generated.append(ind_to_char[predicted_id])

    text=start_seed + "".join(text_generated)
    text_file = open("results.txt", "w")
    n = text_file.write(text)
    text_file.close()

def sparse_cat_loss(y_true, y_pred): #wrapper
  return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True) #for one-hot-encoded features

def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):
  model = Sequential()
  model.add(Embedding(vocab_size, embed_dim, batch_input_shape=[batch_size, None]))
  model.add(GRU(rnn_neurons, 
                stateful=True, return_sequences=True, recurrent_initializer='glorot_uniform'))
  model.add(Dense(vocab_size))

  model.compile(optimizer='adam', loss=sparse_cat_loss) #this had to be wrapped
  
  return model
 
def recover_model(vocab_size=84, embed_dim=64, rnn_neurons=1024+256):
  #my_shakesepeare3 will load with these parameters
  model = create_model(vocab_size, embed_dim, rnn_neurons, batch_size=1)
  model.load_weights("my_shakespeare.h5") #deployment
  model.build(tf.TensorShape([1, None]))

  return model


app = Flask(__name__)
app.config['SECRET_KEY'] = 'shibboleth5'

class SonnetForm(FlaskForm):
	start_seed = TextField("Your contribution")
	gen_size = TextField("Output length")
	temp = TextField("Mayhem")

	submit = SubmitField("wax poetic!")

@app.route("/", methods=['GET', 'POST'])
def index():

	form = SonnetForm()

	if form.validate_on_submit():
		session['start_seed'] = form.start_seed.data
		session['gen_size'] = form.gen_size.data
		session['temp'] = form.temp.data

		return redirect(url_for("prediction"))
	return render_template('home.html', form=form)

text_generator = recover_model()

@app.route('/prediction')
def prediction():
	content = {}
	content['start_seed'] = session['start_seed']
	content['gen_size'] = int(session['gen_size'])
	content['temp'] = float(session['temp'])

	results = generate_text(text_generator, content)
	with open('results.txt', 'r') as f:
		return render_template('prediction.html', results=f.read())

if __name__=='__main__':
	app.run()