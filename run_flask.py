from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from BaseAPI import BaseAPI
from models.ALBERT.ALBERT import ALBERT_API
from models.LSTM.LSTM import LSTM_API
from models.NaiveBayes.interface import NaiveBayes_API

app = Flask(__name__)
api = Api(app)

model_API = {
    "ALBERT": ALBERT_API(),
    "NaiveBayer": None,
    "LSTM": LSTM_API(),
}

class DemoResource(Resource):
    def __init__(self):
        super(DemoResource, self).__init__()
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('model', type=str)
        self.parser.add_argument('text', type=str)

    def post(self):
        args = self.parser.parse_args()
        model = model_API[args["model"]]
        result = model.run_example(args["text"])
        return {"result": result}


api.add_resource(DemoResource, '/demo')

if __name__ == '__main__':
    app.run(debug=True, host="localhost", port=5100, threaded=False)