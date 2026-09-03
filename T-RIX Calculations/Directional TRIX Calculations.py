import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os

# Reading the excel file:
excel_RIX = r"C:\Kshitij stuff\Horizontal Uncertainty Check\Input data\Project_master_file.xlsx"
df_mainframe = pd.read_excel(excel_RIX, header=None)

df_transposed = df_mainframe.T
df_transposed.columns = df_transposed.iloc[0]
df_rixdata = df_transposed[1:]
df_rixdata.reset_index(drop=True, inplace=True)


location = df_rixdata['Location'].tolist()
project = df_rixdata['Project'].tolist()
easting = df_rixdata['Easting'].tolist()
northing = df_rixdata['Northing'].tolist()
z_height = df_rixdata['Z-height'].tolist()

directions = ["N", "NNE", "ENE", "E", "ESE", "SSE",
              "S", "SSW", "WSW", "W", "WNW", "NNW"]
rix_dict = {}
rix_dict = {f"{dir.lower()}_rix": df_rixdata[dir] for dir in directions}
print(n_rix)


