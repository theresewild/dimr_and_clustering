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
from random import sample


#interactive plotting things
# adapted from https://rdkit.blogspot.com/2020/04/new-drawing-options-in-202003-release.html and https://birdlet.github.io/2018/06/06/rdkit_svg_web/
def get_mol_svg(df, smiles_col= None, id_col = None, image_col = None, molSize=(450,150)):
    svgs = []
    if smiles_col == None:
        for col in df.columns:
            if col in ['SMILES', 'smiles', 'Smiles']:
                smiles_col = str(col)
                break
        if smiles_col == None:
            raise ValueError('No default SMILES column found, specify column name using smiles_col = "column_name"')
    smi_col_loc = df.columns.get_loc(smiles_col)
    
    if id_col == None:
        for col in df.columns:
            if col in ['Name', 'LigID', 'id', 'Compound_Name', 'ID']:
                id_col = str(col)
                break
        if id_col == None:
            raise ValueError('No default id column found, specify column name using id_col = "column_name"')
        
    if image_col == None:
        image_col = 'mol_image'

    for i, row in df.iterrows():
        smiles = row[smiles_col]
        try:
            mol = Chem.MolFromSmiles(smiles)
            img = rdMolDraw2D.MolDraw2DSVG(*molSize)
            img.DrawMolecule(mol)
            img.FinishDrawing()
            svg = img.GetDrawingText().replace('svg:', '')
            svgs.append(SVG(svg).data)
        except:
            print (f'error for ID: {row[id_col]}. Please check SMILES: {smiles}')
            
    df.insert(smi_col_loc+1, image_col, svgs)
    return df


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
    

# making plots prettier
def map_alpha_by_col(df, alpha_val_map=None, type_col=None, alpha=0.5):
    if type_col is None:
        type_col_avail = [col for col in ['type', 'class', 'label', 'labels'] if col in df.columns]
        if len(type_col_avail) > 1:
            raise ValueError("Multiple options for type column found, specify column using type_col = 'column_name'")
        elif len(type_col_avail) == 1:
            type_col = type_col_avail[0]
        elif type_col == None:
            raise ValueError("No type column found, specify column using type_col = 'column_name'")

    if alpha_val_map is None:
        alpha_val_map = {}
    
    df['alpha'] = df[type_col].map(alpha_val_map).fillna(alpha)
    return df

def map_color_by_col(df, color_map=None, type_col=None, color='hotpink'):
    if type_col is None:
        type_col_avail = [col for col in ['type', 'class', 'label', 'labels'] if col in df.columns]
        if len(type_col_avail) > 1:
            raise ValueError("Multiple options for type column found, specify column using type_col = 'column_name'")
        elif len(type_col_avail) == 1:
            type_col = type_col_avail[0]
        elif type_col == None:
            raise ValueError("No type column found, specify column using type_col = 'column_name'")

    if color_map is None:
        color_map = {}
    
    df['color'] = df[type_col].map(color_map).fillna(color)
    return df


def get_hopkins_stat(features):
  d = features.shape[1]
  n = len(features) # rows
  m = int(0.1 * n)
  
  nbrs = NearestNeighbors(n_neighbors=1).fit(features.values)
  rand_features = sample(range(0, n, 1), m)
 
  ujd = []
  wjd = []
  for j in range(0, m):
     u_dist, _ = nbrs.kneighbors(uniform(np.amin(features,axis=0),np.amax(features,axis=0),d).reshape(1, -1), 2, return_distance=True)
     ujd.append(u_dist[0][1])
     w_dist, _ = nbrs.kneighbors(features.iloc[rand_features[j]].values.reshape(1, -1), 2, return_distance=True)
     wjd.append(w_dist[0][1])
 
  hopkins_stat = sum(ujd) / (sum(ujd) + sum(wjd))
  if isnan(hopkins_stat):
     print(ujd, wjd)
     hopkins_stat = 0
 
  return hopkins_stat

