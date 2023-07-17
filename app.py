from flask import Flask , render_template,request,flash,redirect,url_for
import tensorflow as tf 
import tensorflow_hub as hub
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
app = Flask(__name__)

upload_folder = 'static/uploads'
pred_folder = 'static/preds'
saved_model_path = 'static/model'
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['PRED_FOLDER'] = pred_folder
app.config['saved_model_path'] = saved_model_path
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
try:
    load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    model = tf.saved_model.load(app.config['saved_model_path'],options=load_options)
except Exception as e:
    model = None
    print(e)
##################helper functions ###############################
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
  """ Loads image from path and preprocesses to make it model ready
      Args:
        image_path: Path to the image file
  """
  hr_image = tf.image.decode_image(tf.io.read_file(image_path))
  # If PNG, remove the alpha channel. The model only supports
  # images with 3 color channels.
  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = tf.cast(hr_image, tf.float32)
  return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
  """
    Saves unscaled Tensor Images.
    Args:
      image: 3D image tensor. [height, width, channels]
      filename: Name of the file to save.
  """
  try:
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save(os.path.join(app.config['PRED_FOLDER'], filename))
  except Exception as e:
      print(e)

######################################################
@app.route('/',methods=['GET','POST'])
def index():
    if model == None:
        flash('Model not Loaded Correctly')
    return render_template('index.html')  

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename) and model != None:
            filename = secure_filename(file.filename)
            ext = filename.split('.')[-1]
            sname = 'Test.'+ str(ext)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], sname))
            img = preprocess_image(os.path.join(app.config['UPLOAD_FOLDER'], sname))
            pred = model(img)
            save_image(tf.squeeze(pred),sname)
            return render_template('pred.html',filename=sname)
        else:
            flash('Error in File or Model')
            return redirect('/')
        

@app.route('/display/<filename>')
def display(filename):
    return redirect(url_for('static',filename='uploads/'+filename),code=301)

@app.route('/displaypred/<filename>')
def displaypred(filename):
    return redirect(url_for('static',filename='preds/'+filename),code=301)

if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=False,host='0.0.0.0',port=5000)