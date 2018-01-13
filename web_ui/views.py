#! /usr/bin/env python3
# coding: utf-8
from flask import Flask, render_template, request, json, jsonify, redirect, url_for
from lxml import html
from server import mef, model
import requests
import sys


app = Flask(__name__)
app.config.from_object('config')

@app.route('/')
@app.route('/index/')
def index():
	return render_template('index.html', numBits = app.config['NUMBITS_PAILLIER_KEYS'], precisionMaxData = app.config['PRECISION_DATA'])

# Serveur chargé de retourner une prédiction une fois les données du patient transmises
@app.route('/server/', methods = ['POST'])
def server():
	content = request.get_json()
	print(content, file=sys.stderr)

	# On crée le fichier du patient
	name = "Fichier du patient n°" + str(content['id'])
	patientFile = mef.MedicalEncryptedFile(name, content['id'], content['encData'])

	# On le transmet au modèle de Machine Learning pour aboutir à une prédiction
	# ml = model.MLModel('', app.config['PRECISION_DATA'])
	# ml.train_model()
	# ml.test_predict()
	
	# ml.predict(patientFile)

	prediction = 1
	page = requests.get(build_url('cypher'), data = {'toEncrypt': prediction,'pubkey':content['pubkey']})
	print(page.content, file=sys.stderr)
	tree = html.fromstring(page.content)
	print(tree, file=sys.stderr)
	encPrediction = tree.xpath('//div[@id="results"]/text()')
	print(encPrediction, file=sys.stderr)
	# encPrediction = 1
	# res.json()

	return jsonify({"status": "done", "id":content['id'], "prediction":encPrediction})


@app.route('/cypher/')
def cypher():
	return render_template('cypher.html', data = request.args.get('data'))

def build_url(page):
    return app.config['DOMAIN'] + page +'/'


if __name__ == "__main__":
	app.run()