import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from rdkit import Chem

from sklearn.neighbors import NearestNeighbors
from math import isnan
from numpy.random import uniform

# import holov# iews as hv
# hv.extension('bokeh')
# import bokeh
from IPython.display import SVG
from rdkit.Chem.Draw import rdMolDraw2D
from bokeh.models import HoverTool
from random import sample

import holoviews as hv
from holoviews import dim
hv.extension('bokeh')

import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


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
            if col in ['Name', 'LigID', 'id', 'Compound_Name', 'ID', 'name']:
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
            svgs.append('no SMILES available')
            
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
    
def map_size_by_col(df, size_map=None, type_col=None, size=100):
    if type_col is None:
        type_col_avail = [col for col in ['type', 'class', 'label', 'labels'] if col in df.columns]
        if len(type_col_avail) > 1:
            raise ValueError("Multiple options for type column found, specify column using type_col = 'column_name'")
        elif len(type_col_avail) == 1:
            type_col = type_col_avail[0]
        elif type_col == None:
            raise ValueError("No type column found, specify column using type_col = 'column_name'")

    if size_map is None:
        size_map = {}
    
    df['size'] = df[type_col].map(size_map).fillna(size)
    return df
    
def map_marker_by_col(df, marker_map=None, type_col=None, marker='o'):
    if type_col is None:
        type_col_avail = [col for col in ['type', 'class', 'label', 'labels'] if col in df.columns]
        if len(type_col_avail) > 1:
            raise ValueError("Multiple options for type column found, specify column using type_col = 'column_name'")
        elif len(type_col_avail) == 1:
            type_col = type_col_avail[0]
        elif type_col == None:
            raise ValueError("No type column found, specify column using type_col = 'column_name'")

    if marker_map is None:
        marker_map = {}
    
    df['marker'] = df[type_col].map(marker_map).fillna(marker)
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
  
  
def plot_umap(features, ID, image, smiles, n_neighbors, min_dist, distance_metric, random_state, 
              font_size=16, type_column=None, alpha=0.7, color='#7B1B79', marker='o', size=100,  
              alpha_value_mapping=None, color_mapping=None, marker_mapping=None, size_mapping=None, plot_order=None,
              export_excel=False, excel_filename="umap_coordinates_df.xlsx",
              save_plot=False, plot_filename="umap_plot.png"):
    
    reducer = umap.UMAP(n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        n_components=2,
                        metric=distance_metric,
                        random_state=random_state)
    
    umap_results = reducer.fit_transform(features)
    umap_df = pd.DataFrame(umap_results, columns=['UMAP1', 'UMAP2'])
    umap_df['name'] = ID
    umap_df['image'] = image
    umap_df['smiles'] = smiles
    
    plt.figure(figsize=(6, 6))
    
    if type_column is not None:
        umap_df['type'] = type_column
        unique_types = sorted(umap_df['type'].unique())  # Ensure consistent ordering
        
        num_types = len(unique_types)  # How many unique categories there are

        default_colors = ['#712377', '#3B8F8F', '#A3C6BE', '#A44660', '#B4C285']
        if len(default_colors) < num_types:
            cmap = cm.get_cmap('viridis', num_types)  # 'tab10' provides distinct colors
            default_colors = [cmap(i) for i in range(num_types)]

        default_markers = ['s', 'o', 'D', '^', 'v', '*', 'X', 'P']
        while len(default_markers) < num_types:
            default_markers.extend(default_markers)  # Repeat pattern

        default_sizes = [50, 80, 100, 120, 140, 160, ]
        while len(default_sizes) < num_types:
            default_sizes.extend(default_sizes)
            
        default_alpha = {t: 0.8 for t in unique_types}

        color_mapping = color_mapping or {t: default_colors[i] for i, t in enumerate(unique_types)}
        marker_mapping = marker_mapping or {t: default_markers[i] for i, t in enumerate(unique_types)}
        size_mapping = size_mapping or {t: default_sizes[i] for i, t in enumerate(unique_types)}
        alpha_value_mapping = alpha_value_mapping or default_alpha

        umap_df = map_alpha_by_col(umap_df, alpha_value_mapping, alpha=1)
        umap_df = map_color_by_col(umap_df, color_mapping)
        umap_df = map_marker_by_col(umap_df, marker_mapping)
        umap_df = map_size_by_col(umap_df, size_mapping)

        plot_order = plot_order or unique_types  # Default to plotting all types
        
        for t in plot_order:
            subset = umap_df[umap_df['type'] == t]
            plt.scatter(subset['UMAP1'], subset['UMAP2'], 
                        alpha=subset['alpha'].iloc[0], 
                        color=subset['color'].iloc[0], 
                        marker=subset['marker'].iloc[0], 
                        s=subset['size'].iloc[0], label=t)
        
        plt.legend(title='Type', fontsize=12)
    
    else:
        plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'], alpha=alpha, color=color, marker=marker, s=size)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)

    plt.xlabel('UMAP1', fontsize=16)
    plt.ylabel('UMAP2', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    if save_plot:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {plot_filename}")
        
    if export_excel:
        umap_df.to_excel(excel_filename, index=False)
        print(f"UMAP data saved to {excel_filename}")
    
    plt.show()
    
    return umap_df


def plot_tsne(features, ID, image, smiles, perplexity, early_exaggeration,learning_rate, distance_metric, random_state, init='random',
              font_size=16, type_column=None, alpha=0.7, color='#7B1B79', marker='o', size=100,  
              alpha_value_mapping=None, color_mapping=None, marker_mapping=None, size_mapping=None, plot_order=None,
            export_excel=False, excel_filename="tsne_coordinates_df.xlsx",
            save_plot=False, plot_filename="tsne_plot.png"):
              
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, metric=distance_metric, 
            init=init, random_state=random_state, )
    tsne_results = tsne.fit_transform(features)

    tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    tsne_df['name'] = ID
    tsne_df['image'] = image
    tsne_df['smiles'] = smiles
    
    plt.figure(figsize=(6, 6))
    
    if type_column is not None:
        tsne_df['type'] = type_column
        unique_types = sorted(tsne_df['type'].unique())  # Ensure consistent ordering
        
        num_types = len(unique_types)  # How many unique categories there are

        default_colors = ['#712377', '#3B8F8F', '#A3C6BE', '#A44660', '#B4C285']
        if len(default_colors) < num_types:
            cmap = cm.get_cmap('viridis', num_types)  # 'tab10' provides distinct colors
            default_colors = [cmap(i) for i in range(num_types)]

        default_markers = ['s', 'o', 'D', '^', 'v', '*', 'X', 'P']
        while len(default_markers) < num_types:
            default_markers.extend(default_markers)  # Repeat pattern

        default_sizes = [50, 80, 100, 120, 140, 160, ]
        while len(default_sizes) < num_types:
            default_sizes.extend(default_sizes)
            
        default_alpha = {t: 0.8 for t in unique_types}

        color_mapping = color_mapping or {t: default_colors[i] for i, t in enumerate(unique_types)}
        marker_mapping = marker_mapping or {t: default_markers[i] for i, t in enumerate(unique_types)}
        size_mapping = size_mapping or {t: default_sizes[i] for i, t in enumerate(unique_types)}
        alpha_value_mapping = alpha_value_mapping or default_alpha

        tsne_df = map_alpha_by_col(tsne_df, alpha_value_mapping, alpha=1)
        tsne_df = map_color_by_col(tsne_df, color_mapping)
        tsne_df = map_marker_by_col(tsne_df, marker_mapping)
        tsne_df = map_size_by_col(tsne_df, size_mapping)

        plot_order = plot_order or unique_types  # Default to plotting all types
        
        for t in plot_order:
            subset = tsne_df[tsne_df['type'] == t]
            plt.scatter(subset['TSNE1'], subset['TSNE2'], 
                        alpha=subset['alpha'].iloc[0], 
                        color=subset['color'].iloc[0], 
                        marker=subset['marker'].iloc[0], 
                        s=subset['size'].iloc[0], label=t)
        
        plt.legend(title='Type', fontsize=12)
    
    else:
        plt.scatter(tsne_df['TSNE1'], tsne_df['TSNE2'], alpha=alpha, color=color, marker=marker, s=size)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)

    plt.xlabel('tsne1', fontsize=16)
    plt.ylabel('tsne2', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    if save_plot:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {plot_filename}")
        
    if export_excel:
        tsne_df.to_excel(excel_filename, index=False)
        print(f"tsne data saved to {excel_filename}")
    
    plt.show()
    
    return tsne_df

def plot_pca(features, ID, image, smiles, n_components=2, alpha=0.7, color='#7B1B79', marker='o', size=100,  
             alpha_value_mapping=None, color_mapping=None, marker_mapping=None, size_mapping=None, plot_order=None,
             pc_x='PC1', pc_y='PC2', pc_z=None, font_size=16, type_column=None, 
             export_excel=False, excel_filename="pca_coordinates_df.xlsx",
             save_plot=False, plot_filename="pca_plot.png"):
    
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(features)
    pca_score = pca.explained_variance_ratio_
    
    print('Total variance explained by PCs:', round(np.sum(pca_score * 100), 1), '%\n')
    print("Percentage of explained variance per principal component:")
    for i, j in enumerate(pca_score):
        print(f"PC{i+1}   {j * 100:.1f}%")
    
    pca_df = pd.DataFrame(pca_results, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df['name'] = ID
    pca_df['image'] = image
    pca_df['smiles'] = smiles
    
    if type_column is not None:
        pca_df['type'] = type_column
        unique_types = sorted(pca_df['type'].unique())
        
        
        default_colors = ['#712377', '#3B8F8F', '#A3C6BE', '#A44660', '#B4C285']
        if len(default_colors) < len(unique_types):
            cmap = cm.get_cmap('viridis', unique_types) 
            default_colors = [cmap(i) for i in range(unique_types)]
        
        default_markers = ['s', 'o', 'D', '^', 'v', '*', 'X', 'P']
        while len(default_markers) < len(unique_types):
            default_markers.extend(default_markers)
        
        default_sizes = [50, 80, 100, 120, 140, 160, ]
        while len(default_sizes) < len(unique_types):
            default_sizes.extend(default_sizes)
            
        default_alpha = {t: 0.8 for t in unique_types}

        color_mapping = color_mapping or {t: default_colors[i] for i, t in enumerate(unique_types)}
        marker_mapping = marker_mapping or {t: default_markers[i] for i, t in enumerate(unique_types)}
        size_mapping = size_mapping or {t: default_sizes[i] for i, t in enumerate(unique_types)}
        alpha_value_mapping = alpha_value_mapping or default_alpha
        
        pca_df = map_alpha_by_col(pca_df, alpha_value_mapping, alpha=1)
        pca_df = map_color_by_col(pca_df, color_mapping)
        pca_df = map_marker_by_col(pca_df, marker_mapping)
        pca_df = map_size_by_col(pca_df, size_mapping)
        
        plot_order = plot_order or unique_types  # Default to plotting all types
        
    else:
        pca_df['alpha'] = alpha
        pca_df['color'] = color
        pca_df['marker'] = marker
        pca_df['size'] = size
        plot_order = [None]  # Only one category to plot
    
    fig = plt.figure(figsize=(6, 6))
    
    if pc_z is None:
        ax = plt.gca()  # 2D plot
        for t in plot_order:
            subset = pca_df[pca_df['type'] == t] if t else pca_df
            ax.scatter(subset[pc_x], subset[pc_y], 
                       alpha=subset['alpha'].iloc[0], 
                       color=subset['color'].iloc[0], 
                       marker=subset['marker'].iloc[0], 
                       s=subset['size'].iloc[0], label=t)
        ax.set_xlabel(pc_x, fontsize=font_size)
        ax.set_ylabel(pc_y, fontsize=font_size)
    
    else:
        ax = fig.add_subplot(111, projection='3d')  # 3D plot
        for t in plot_order:
            subset = pca_df[pca_df['type'] == t] if t else pca_df
            ax.scatter(subset[pc_x], subset[pc_y], subset[pc_z], 
                       alpha=subset['alpha'].iloc[0], 
                       color=subset['color'].iloc[0], 
                       marker=subset['marker'].iloc[0], 
                       s=subset['size'].iloc[0], label=t)
        ax.set_xlabel(pc_x, fontsize=font_size)
        ax.set_ylabel(pc_y, fontsize=font_size)
        ax.set_zlabel(pc_z, fontsize=font_size)
    
    ax.tick_params(labelsize=font_size)
    plt.legend(title='Type', fontsize=12)
    
    # Saving the plot and/or exporting the DataFrame
    if save_plot:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {plot_filename}")
    
    if export_excel:
        pca_df.to_excel(excel_filename, index=False)
        print(f"PCA data saved to {excel_filename}")
    
    plt.show()
    
    return pca_df


