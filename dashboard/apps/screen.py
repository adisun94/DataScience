from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from glob import glob
import numpy as np
import pandas as pd
import pathlib
from app import app

import sys
sys.path.append('/home/adisun/DataScience/')
from Functions import ChemClean
from Functions import Descriptors

import rdkit.Chem as Chem
from rdkit.Chem import rdMolDescriptors 
from sklearn.preprocessing import StandardScaler
import pickle

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath().resolve()

df=pd.read_csv(DATA_PATH.joinpath('../data/MolRedPot_data.csv'))
df.insert(0,'id',list(range(df.shape[0])))
df.drop(columns=['ERed'],inplace=True)
pot=np.loadtxt(DATA_PATH.joinpath('../../GBR_predictions.txt'))

layout = html.Div([
    dcc.Markdown('''
        #### This dashboard lists 5442 molecules, whose reduction potentials are predicted using a Gradient Boosting Regression model. The code repository is available [here](https://github.com/adisun94/DataScience) and [here](https://github.com/akashjn/DataScience).'''),
    html.Br(),
    dash_table.DataTable(
        id='table-smiles',
        columns=[
            {'name':iname,'id':iname,'selectable':True}
            for iname in df.columns
        ],
        data=df.to_dict('records'),
        export_format='csv',
        editable=False,
        filter_action='native',
        sort_action='native',
        sort_mode='single',
        row_selectable=False,
        page_action='native',
        page_current=0,
        page_size=10,
        style_cell={
            'minWidth':25, 'maxWidth':25, 'width':25,'fontSize':13
        },
        style_header_conditional=[
            {"if": {"column_id": "SMILES"}, "color": "blue",'fontWeight': 'bold'},
            {"if": {"column_id": "ERed"}, "color": "red",'fontWeight': 'bold'},
            ],
        style_data={'color': 'black','backgroundColor': 'white'},
        style_data_conditional=[{'if': {'row_index': 'odd'},'backgroundColor': 'rgb(245, 245, 245)',}],
        ),
    html.Div([
        "Enter molecule id between 0 and 5441 and press 'Enter': ",
        dcc.Input(id='moleculeID', value=123, type='number',placeholder="Molecule ID", debounce=True,min=0, max=1183, step=1)
        ]),
    html.Br(),
    html.Div(id='smiles'),
    html.Br(),
    html.Div(id='potential'),
    dcc.Graph(id='PD', figure={})
    ])

@app.callback(
    Output(component_id='smiles',component_property='children'),
    Output(component_id='potential', component_property='children'),
    Output(component_id='PD', component_property='figure'),
    Input(component_id='moleculeID', component_property='value')
    )

def update_output(input_id):

    print(input_id)

    fig=make_subplots()
    x,y=list(range(10)),list(range(10))
    #fig.add_trace(go.Scatter(x=x,y=y))

    #fig.update_xaxes(title='Temperature (C)')
    #fig.update_yaxes(title='Mass fraction')
    #fig.update_layout(height=600, width=600)
    #fig.update_layout({'paper_bgcolor': 'rgb(255,255,240)'})

    o1='The molecule id selected is: '+str(input_id)+'\n'+'The molecule SMILES string is: '+df['SMILES'][input_id]
    o2='the predicted reduction potential = '+str(pot[input_id])+' V'
    return o1,o2,fig

