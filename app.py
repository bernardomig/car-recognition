from flask import Flask, request, send_file, Response
import base64
from StringIO import StringIO
from PIL import Image
import numpy as np
import json
import random

import matplotlib.pyplot as plt

app = Flask(__name__,
            static_url_path='',
            static_folder='static',
            template_folder='templates')


@app.route('/', methods=['GET'])
def index():
    return send_file('index.html')


@app.route('/classify', methods=['POST'])
def classify():

    data = request.form['img']

    content = data.split(';')[1]
    image_encoded = content.split(',')[1]
    img = base64.decodestring(image_encoded.encode('utf-8'))

    img = Image.open(StringIO(img))

    # 224x224x3

    # TODO: Insert classifier here

    x = random.randrange(0, 100)
    y = random.randrange(0, 100)
    w = random.randrange(50, 200)
    h = random.randrange(50, 200)

    bbx = [
        {'x': x,
         'y': y,
         'w': w,
         'h': h}
    ]

    return Response(json.dumps({'bbx': bbx}), mimetype=u'application/json')
