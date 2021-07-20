from flask import Flask, jsonify
from flask_cors import CORS

from routes.route import create_route
from models.model import load
from utils.utilities import ErrorAPI

app = Flask(__name__)
CORS(app)


@app.errorhandler(ErrorAPI)
def exception(e: ErrorAPI):
    return jsonify(e.detail()), 200


@app.errorhandler(Exception)
def exception(e: Exception):
    return jsonify({'error': str(e)}), 500


models = load()


bp = create_route(models)
CORS(bp)
app.register_blueprint(bp, url_prefix='/api')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
