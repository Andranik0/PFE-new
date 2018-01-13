#! /usr/bin/env python3
# coding: utf-8
from web_ui import app
import os

port = int(os.environ.get('PORT', 8080))

if __name__ == "__main__":
    app.run(debug=True, port=port)