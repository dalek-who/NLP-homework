from flask import Blueprint,send_file,Flask,request,jsonify
from flask_restful import Resource, Api, reqparse
from BaseAPI import BaseAPI
from models.ALBERT.interface import ALBERT_API
from models.LSTM.interface import LSTM_API
from models.NaiveBayes.interface import NaiveBayes_API

app = Flask(__name__)
api = Api(app)

model_API = {
    "ALBERT": ALBERT_API(),
    "NaiveBayes": NaiveBayes_API(),
    "LSTM": LSTM_API(),
}

label_dict = {
    0: "正面",
    1: "中性",
    2: "负面"
}
# class DemoResource(Resource):
#     def __init__(self):
#         super(DemoResource, self).__init__()
#         self.parser = reqparse.RequestParser()
#         self.parser.add_argument('model', type=str)
#         self.parser.add_argument('text', type=str)
#
#     def post(self):
#         args = self.parser.parse_args()
#         model = model_API[args["model"]]
#         result = model.run_example(args["text"])
#         return {"result": label_dict[result]}
@app.route("/")
def demo():
    print("demo")
    return send_file("static/demo.html")

@app.route('/demo', methods=['POST'])
def demo1():
    req_dict = request.get_json()
    print(req_dict)
    model = req_dict.get("model")
    text = req_dict.get("text")
    model = model_API[model]
    result = model.run_example(text)
    result_dict =label_dict[result]
    print(result_dict)
    return jsonify({"result":result_dict})
# api.add_resource(DemoResource, '/demo')

if __name__ == '__main__':
    app.run(debug=True, host="localhost", port=5000, threaded=False)