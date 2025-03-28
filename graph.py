import torch
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json


class Grapher:
    def __init__(self, path):
        self.paths = [path]
        # args = pd.read_csv(f'{path}/args.log', index_col=0, keep_default_na=False, na_values=['NaN'])
        # train = pd.read_csv(f'{path}/train.log')
        # val = pd.read_csv(f'{path}/eval.log')
        # args = pd.read_csv(os.path.join(path, 'args.json'), index_col=0, keep_default_na=False, na_values=['NaN'])
        with open(os.path.join(path, 'args.json')) as f:
            args_json = json.load(f)
        args2 = {}
        # for key, value in args.items():
        #     if isinstance(value, dict):
        #         for k, v in value.items():
        #             args2[key+'_'+k] = v
        #     else:
        #         args2[key] = value
        args2 = args_json['model_args']
        args2['pos_enc'] = args_json['args']['position_encoding']
        args = pd.DataFrame(args2, index=[0]).T
        args.columns = ['value']
        

        train = pd.read_csv(os.path.join(path, 'train.log'))
        val = pd.read_csv(os.path.join(path, 'eval.log'))
        

        # Replace nan with none

        train['perplexity'] = 2**(train['loss'])
        val['perplexity'] = 2**(val['loss'])
        train['perplexity'] = np.exp(train['loss'])
        val['perplexity'] = np.exp(val['loss'])

        train['tokens'] = train['step']*args_json['dataset_args']['tokens_per_batch']
        val['tokens'] = val['step']*args_json['dataset_args']['tokens_per_batch']


        self.diff = set()
        self.args = [args]
        self.train_dfs= [train]
        self.val_dfs= [val]

        self.type_list = []
    
    def __add__(self, other):
        self.paths.extend(other.paths)
        self.args.extend(other.args)
        self.train_dfs.extend(other.train_dfs)
        self.val_dfs.extend(other.val_dfs)
        self.diff = self.args_diff()
        return self
    
    def __repr__(self):
        # px.line(self.train, x='step', y='loss', title='Train Loss', color='type').show()
        self.plot().show()
        return ''
    
    def plot(self, x='step', y='loss', plot_set='all', do_buttons=True, range_y=None, range_x=None):
        self.set_types()
        title = f'{y[0].upper() + y[1:]} by {x[0].upper() + x[1:]}'
        if plot_set == 'all':
            df_list = []
            for train, val in zip(self.train_dfs, self.val_dfs):
                df_list.append(train)
                df_list.append(val)
            df = pd.concat(df_list)
        elif plot_set == 'train':
            df = pd.concat(self.train_dfs)
            df['type'] = df['type'].str.removeprefix('train_')
        elif plot_set == 'val':
            df = pd.concat(self.val_dfs)
            df['type'] = df['type'].str.removeprefix('val_')
        else:
            raise ValueError('plot_set must be either all, train, or val')
        df['Type'] = df['type']
        df['Time'] = df['time']/3600
        fig = px.line(df, x=x, y=y, title=title, color='Type', range_y=range_y, range_x=range_x)

        if do_buttons:
            buttons = []
            # "All" button (show all)
            buttons.append(dict(
                label="All",
                method="update",
                args=[{"visible": True},  # Show all traces
                    {"title": "Scatter Plot with Color Toggle (All Colors)", "showlegend": True,  "updatemenus[1].active": 0}]
            ))

            # Individual category buttons (including hiding all others)
            for category in ['train', 'val']:
                # visible = [True if trace.name == category else False for trace in fig.data]  # Show only traces matching the category
                visible = [True if category in trace.name else False for trace in fig.data]  # Show only traces matching the category
                buttons.append(dict(
                    label=category,
                    method="update",
                    args=[{"visible": visible},
                        {"title": f"Scatter Plot (Showing only: {category})",  "showlegend": True,  "updatemenus[0].active": 0}]
                ))


            dropdown_buttons = []
            # "All" option for the dropdown
            dropdown_buttons.append(dict(
                label="All",
                method="update",
                args=[{"visible": True},
                    {"title": "Scatter Plot (All Splits)", "showlegend": True}]
            ))

            for split_cat in self.type_list:
                visible = [split_cat in trace.name for trace in fig.data]
                dropdown_buttons.append(dict(
                    label=split_cat,
                    method="update",
                    args=[{"visible": visible},
                        {"title": f"Scatter Plot (Showing only: {split_cat})", "showlegend": True}]
                ))

            fig.update_layout(
                margin=dict(l=50, r=150, t=100, b=50),
                updatemenus=[
                    dict(
                        type="dropdown",
                        direction="down",
                        x=1.05,
                        # y=1,
                        y=1.3,
                        xanchor="left",
                        yanchor="top",
                        showactive=True,
                        buttons=buttons,
                        pad={"r": 10, "t": 10},
                    ),
                    dict(
                        type="dropdown",
                        direction="down",
                        x=1.05,
                        y=1.15,
                        # y=0.85,
                        xanchor="left",
                        yanchor="top",
                        showactive=True,
                        buttons=dropdown_buttons,
                        pad={"r": 10, "t": 5},
                    )
                ],
                showlegend=True,  # Show the legend *outside* the plot area
                legend=dict(
                    x=1.05,  # Position the legend to the right
                    y=1,      # Adjust vertical position as needed
                    # y=0.7,      # Adjust vertical position as needed
                    xanchor="left", #anchor for legend
                    yanchor="top",  #anchor for legend
                )
            )
        return fig
    


    def __call__(self, x='step', y='loss', plot_set='all'):
        return self.plot(x, y, plot_set)            
    
    def args_diff(self):
        args_intersection = set(self.args[0].index)
        for i in range(1, len(self.args)):
            args_intersection = args_intersection.intersection(self.args[i].index)
        diff = []
        for arg in args_intersection:
            if not all(self.args[0].loc[arg].value == self.args[i].loc[arg].value for i in range(1, len(self.args))):
                diff.append(arg)
        return diff
            
    
    def set_types(self):
        # self.type_list = []
        for i in range(len(self.args)):
            self.train_dfs[i]['type'] = 'train'
            self.val_dfs[i]['type'] = 'val'
            for key in self.diff:
                self.train_dfs[i]['type'] += f'_{self.args[i].loc[key].value}'
                self.val_dfs[i]['type'] += f'_{self.args[i].loc[key].value}'
                # self.type_list.append(self.args[i].loc[key].value)
        # Check if there are any duplicates, if so, add and id to them
        for i in range(len(self.args)):
            id = 1
            for j in range(i+1, len(self.args)):
                if self.train_dfs[i]['type'].loc[0] == self.train_dfs[j]['type'].loc[0]:
                    self.train_dfs[j]['type'] += f'_{id}'
                    self.val_dfs[j]['type'] += f'_{id}'
                    id += 1
            if id > 1:
                self.train_dfs[i]['type'] += f'_0'
                self.val_dfs[i]['type'] += f'_0'
        self.type_list = []
        for i in range(len(self.args)):
            self.type_list.extend(self.train_dfs[i]['type'].unique())
        self.type_list = list(set(self.type_list))

    def save_fig(self, x='step', y='perplexity', plot_set='val', range_y=None, range_x=None, shape=None):
        fig = self.plot(x, y, plot_set, do_buttons=False, range_y=range_y, range_x=range_x)

        if shape:
            # Shape is a tuple of (height, width)
            fig.update_layout(
                autosize=True,
                width=shape[1],
                height=shape[0],
            )

        # Set background color to transparent rgba(0,0,0,0)
        fig.update_layout(      
            legend=dict(
                x=.8,
                y=1,
                traceorder="normal",
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="black",
                ),
                bgcolor="rgba(0,0,0,0)",
            )
        )   
        # fig.show()
        os.makedirs('images', exist_ok=True)
        filepath = os.path.join('images', f'{y}_by_{x}.png')
        fig.write_image(filepath, scale=4)
        # if show:
        #     fig.show()
        return fig
        
