import pandas as pd
from flask import Flask, request, render_template #render_template help in getting back to home page.
import pickle

# First preprocessing text.
# second count vectorizing.
# Third Finding what language is given for that ouput label.

label_encoder = pickle.load(open('models/labelencoder.pkl','rb'))
model = pickle.load(open('models/Language_Detection_model.pkl','rb'))

app = Flask(__name__, template_folder='templates')

def predictions(data_dict):
     df = pd.DataFrame(data_dict)
     pred = model.predict(df)
     print(pred,'\n',pred[0])
     return pred[0]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
     if request.method == 'POST':
          to_predict_list = request.form.to_dict()
          Text = int(to_predict_list['Text'])
          print(Text)

          data_dict = {
          'Text': [Text],
          }

          res = predictions(data_dict)
          result = res.apply(lambda x: label_encoder.fit_transform(x))

     return render_template('index.html',
                            textlan = Text,
                            Result=result)

if __name__=="__main__":
     app.run(debug=True,host="0.0.0.0",port="8080")





     X = []
for i in range (len(data)):
  sentence = data['Text'][i]
  translation_table = str.maketrans("", "", string.punctuation)
  cleaned_sentence = sentence.translate(translation_table)
  lowercase_string = cleaned_sentence.lower()
  X.append(lowercase_string)