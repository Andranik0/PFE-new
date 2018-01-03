#! /usr/bin/env python3
# coding: utf-8
from flask import Flask, render_template, request, jsonify
from server import mef, model
from server import uploadHandler as uh
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
	test = uh.UploadHandler()
	# id = request.json['id']
	# data = request.json['encData']
	# pubkey = request.json['pubkey']

	# On crée le fichier du patient
	# name = "Fichier du patient n°" + str(content['id'])
	# patientFile = mef.MedicalEncryptedFile(name,content['id'],content['encData'])

	# On le transmet au modèle de Machine Learning pour aboutir à une prédiction
	# ml = model.MLModel('', app.config['PRECISION_DATA'])
	# ml.train_model()
	# ml.test_predict()
	print(content, file=sys.stderr)
	# ml.predict(patientFile)

	# return jsonify({"status": "ok", "id":content.id})
	return jsonify({"status": "ok", "id":0})

if __name__ == "__main__":
	app.run()