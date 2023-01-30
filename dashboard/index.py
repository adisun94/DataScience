from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from skimage import io
from app import app
import plotly.express as px

data_path = 'dashboard_data_5442mols.csv'

df = pd.read_csv(data_path, header=0, index_col=0)

app.layout = html.Div([

    html.Div(children=[
        dcc.Markdown('''
                #### This dashboard lists 5442 molecules, whose reduction potentials are predicted using a Gradient Boosting Regression model. The code repository including data, model pipelines is available [here](https://github.com/adisun94/DataScience) and [here](https://github.com/akashjn/DataScience). More information about Redox Flow Batteries is available [here](https://energystorage.org/why-energy-storage/technologies/redox-flow-batteries/)''')
        ],
        style={'padding': 10, 'flex': 1, 'background-color' : 'rgb(255,255,240)'}),
    html.Br(),
    html.Div(children=[
        html.Label(['Select colorbar property'], style={'font-weight': 'bold', "text-align": "left"}),
        dcc.Dropdown(
            df.columns.drop(['URL']),
            value='MolWT',
            id='property',
            clearable=False)
    ]   ,
        style={"width": "20%"}),
    html.Br(),
    html.Div(children=[
        dcc.Graph(
            id='scatterplot',
            figure={}
        ),
        dcc.Graph(
            id='moleculeimage',
            figure={}
        )],
        style={'display': 'flex', 'flex-direction': 'row', 'background-color' : 'rgb(255,255,240)'})
    ])
    
@app.callback(
    Output('scatterplot','figure'),
    Input('property','value')
)

def scatter_plot(value):

    fig = go.Figure(data=[
    go.Scatter(
        x=df["MolLogP"],
        y=df["ERed"],
        mode="markers",
        marker=dict(
            colorscale='viridis',
            color=df[value],
            size=df["MolWT"],
            colorbar={"title": value+' (units)'},
            line={"color": "#444"},
            reversescale=True,
            sizeref=45,
            sizemode="diameter",
            opacity=0.8,
            )
        )])
       
    fig.update_layout(width=1100,height=650)

    fig.update_traces(hoverinfo="none", hovertemplate=None)

    fig.update_layout(
        title='Molecule dataset : Redoxmer electrolytes for Redox Flow Batteries (RFB)',
        xaxis=dict(title='MolLogP'),
        yaxis=dict(title='ERed (V)'),
        font=dict(family="Arial",size=18),
        plot_bgcolor='rgba(220,220,220,1)',
        paper_bgcolor='rgba(255,255,240,1)'
        )

    return fig

@app.callback(
    Output('moleculeimage','figure'),
    Input('scatterplot','hoverData'),
    Input('property','value')
)

def molecule_image(hoverData,value):

    if hoverData is None:

        img = io.imread('assets/molecule.jpg')
        fig = px.imshow(img)

        fig.update_layout(width=400,height=400)

        fig.update_traces(hoverinfo="none", hovertemplate=None)

        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False),

        fig.update_layout(
            plot_bgcolor='rgba(255,255,240,1)',
            paper_bgcolor='rgba(255,255,240,1)',
            margin=dict(l=20, r=20, t=150, b=00)
            )

        return fig

    pt = hoverData["points"][0]
    num = pt["pointNumber"]

    df_row = df.iloc[num]
    img_src=io.imread('assets/molID_'+str(num)+'.png')

    fig = px.imshow(img_src)

    fig.update_layout(width=500,height=500)

    fig.update_traces(hoverinfo="none", hovertemplate=None)

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False),

    fig.update_layout(
        title='<br>Molecule = '+df.index[num]+'<br>'+str(value)+' = '+str(df[value][num]),
        font=dict(family="Arial",size=18),
        plot_bgcolor='rgba(255,255,240,1)',
        paper_bgcolor='rgba(255,255,240,1)',
        margin=dict(l=20, r=20, t=100, b=0)
        )

    return fig

if __name__ == "__main__":
    app.run_server(debug=True)