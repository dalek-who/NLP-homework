from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from BaseModel import BaseModel

app = Flask(__name__)
api = Api(app)

record = []

class DemoResource(Resource):
    def __init__(self):
        super(DemoResource, self).__init__()
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('model', type=str)
        self.parser.add_argument('text', type=str)

    def post(self):
        args = self.parser.parse_args()
        record.append((args["model"], args["text"]))
        return {"result": 1}


api.add_resource(DemoResource, '/demo/')

if __name__ == '__main__':
    app.run(debug=True, host="localhost", port=5100)