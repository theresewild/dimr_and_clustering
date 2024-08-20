import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem

from sklearn.neighbors import NearestNeighbors
from math import isnan
from numpy.random import uniform

import holoviews as hv
hv.extension('bokeh')
import bokeh
from IPython.display import SVG
from rdkit.Chem.Draw import rdMolDraw2D
from bokeh.models import HoverTool
from bokeh.resources import INLINE
from yellowbrick.cluster import SilhouetteVisualizer
from bokeh.io import show






# adapted from https://birdlet.github.io/2018/06/06/rdkit_svg_web/
def DrawMol(dataframe, smiles_column_loc, image_column, id_column='Compound_Name', molSize=(200, 100), kekulize=True):
    images = []
    for index, row in dataframe.iterrows():
        smiles_string = row.iloc[smiles_column_loc]  # Use iloc for position-based indexing
        
        # Check if the SMILES string is blank or NaN
        if pd.isnull(smiles_string) or smiles_string == '':
            print(f"Skipping ID: {row[id_column]} due to blank or NaN SMILES")
            continue

        try:
            mc = Chem.MolFromSmiles(smiles_string)
            if kekulize:
                try:
                    Chem.Kekulize(mc)
                except:
                    mc = Chem.Mol(smiles_string.ToBinary())

            if not mc.GetNumConformers():
                Chem.rdDepictor.Compute2DCoords(mc)

            drawer = rdMolDraw2D.MolDraw2DSVG(*molSize)
            drawer.DrawMolecule(mc)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText().replace('svg:', '')
            images.append(SVG(svg).data)
        except:
            print(f"Error for ID: {row[id_column]} with SMILES: {smiles_string}")

    try:
        dataframe.insert(smiles_column_loc+1, image_column, images)
    except: 
        dataframe[image_column] = images
        
    return dataframe

def get_hopkins_stat(X):
  d = X.shape[1]
  n = len(X) # rows
  m = int(0.1 * n)
  
  nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
  rand_X = sample(range(0, n, 1), m)
 
  ujd = []
  wjd = []
  for j in range(0, m):
     u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
     ujd.append(u_dist[0][1])
     w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
     wjd.append(w_dist[0][1])
 
  H = sum(ujd) / (sum(ujd) + sum(wjd))
  if isnan(H):
     print(ujd, wjd)
     H = 0
 
  return H

def scatter_plot(
    dataframe: pd.DataFrame, 
    x: str,  # x-axis data
    y: str,  # y-axis data
    title: str = 'default',  # title of plot
    x_label: str = 'default',  # axis label to be printed on plot (does not need to match dataframe name)
    x_range: tuple = None,  # range of x-axis
    y_label: str = 'default',  # axis label to be printed on plot (does not need to match dataframe name)
    y_range: tuple = None,  # range of y-axis
    legend: str = '',  # string with data label if using classifiers/building plots by category
    svgs: str = None,  # string with column name of svgs 
    hover_list: list = None,  # list of column names with data to be shown on hover 
    marker: str = 'o',  # marker type - most of the matplotlib markers are supported (https://matplotlib.org/stable/api/markers_api.html)
    color: str = '#931319',  # color of markers
    line_color: str = None,  # color of marker line_color
    alpha: int = 1,  # transparency of markers
    groupby: str = None,  # string with column name to group data by
    height: int = 500,  #plot height (recommended: 500)
    width: int = 500,  #plot width (recommended: 500)
    size: int = 10,  # size of markers (recommended: 10-20)
):

    if x_label == 'default':  # if no x_label provided, use x column name
        x_label = x
    if y_label == 'default':  # if no y_label provided, use y column name
        y_label = y

    if not x_range:
        x_min = min(dataframe[x]); x_max = max(dataframe[x])
        x_buffer = abs(x_max-x_min)/10
        x_range = (x_min-x_buffer, x_max+x_buffer)
    if not y_range:
        y_min = min(dataframe[y]); y_max = max(dataframe[y])
        y_buffer = abs(y_max-y_min)/10
        y_range = (y_min-y_buffer, y_max+y_buffer)

    if groupby is not None:
        hover_list = hover_list or []
        hover_list.insert(0, groupby)

    if svgs == None and labels == None: # no hover information provided
        if title == 'default':  # if no title provided, define from x, y labels
            title = f'{y_label} vs. {x_label}'
        plt = hv.Scatter(dataframe, kdims=[x], vdims=[y], label=legend).opts(title=title, marker=marker, height=height, width=width, color=color, alpha=alpha, size=size, line_color=line_color)
        
    else:  # hover information provided, build list of hover tools
        hover_list.insert(0, y)
        tooltips = f'<div>end' # beginning of tooltips if no svgs provided
        if svgs != None:
            tooltips = f'<div><div>@{svgs}{{safe}}</div>end'  # beginning of tooltips if svgs are provided
            hover_list.insert(1, svgs)
        if len(hover_list) < 4:
            for label in hover_list:
                if label != svgs and label != y:
                    tooltips = tooltips.replace('end', f'<div><span style="font-size: 17px; font-weight: bold;">@{label}</span></div>end')
        else:
            for label in hover_list:
                if label != svgs and label != y:
                    tooltips = tooltips.replace('end', f'<div><span style="font-size: 12px;">{label}: @{label}</span></div>end')
        
        tooltips = tooltips.replace('end', '</div>')
        hover = HoverTool(tooltips=tooltips)
        
        if title == 'default':  # if no title provided, define from x, y labels
            title = f'{y_label} vs. {x_label}'          
        plt = hv.Scatter(dataframe, kdims=[x], vdims=hover_list, label=legend).opts(title=title, marker=marker, height=height, width=width, tools=[hover], color=color, alpha=alpha, size=size, line_color=line_color)
        
        if groupby != None:
            # color = hv.Cycle(color).values
            plt = plt.opts(color=groupby, cmap=color)

        return plt