#! /usr/bin/env python3
# coding: utf-8
import os
import pandas as pd

class MedicalEncryptedFile:
    def __init__(self, name, id, data):
        self._name = name
        self._id = id
        self._data = data

    # def data_from_csv(self, csv_file):
    #     self.dataframe = pd.read_csv(csv_file, sep=",")

    # def data_from_dataframe(self, dataframe):
    #     self.dataframe = dataframe

    def get_name(self):
        return self._name

    def get_id(self):
        return self._id
  
    def get_data(self):
        return self._data

    def predict(self):
        pass

# def launch_analysis(data_file):
#     mf.data_from_csv(os.path.join("data", data_file))

#     with pd.option_context('display.max_rows', 1, 'display.max_columns', None):
#     	print(mf.dataframe)

# if __name__ == '__main__':
# 	launch_analysis("patient_855563_file.csv")