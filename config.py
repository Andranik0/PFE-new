#! /usr/bin/env python3
# coding: utf-8
import os

# Taille des clés privées et publiques générées par Paillier
NUMBITS_PAILLIER_KEYS = 16

# Les données seront systématiquement présentées sous la forme de décimaux arrondis au PRECISION_DATA-ième chiffre après la virgule
PRECISION_DATA = 4

# Domaine du site
if os.environ.get('DOMAIN') is None:
	DOMAIN = 'http://127.0.0.1:8080/'
else:
	DOMAIN = os.environ.get('DOMAIN')