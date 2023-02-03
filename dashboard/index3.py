from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from skimage import io
from app import app
import pickle
import plotly.express as px
import tensorflow as tf

data_path = 'dashboard_data_5442mols.csv'

df = pd.read_csv(data_path, header=0, index_col=None)

lr_model = pickle.load(open('../Models/LinearRegression.sav', 'rb'))
gbr_model = pickle.load(open('../Models/EnsembleGBR.sav', 'rb'))
nn_model = tf.keras.models.load_model('../Models/NN_model_tuned/')

target_train = pd.read_csv('../Data/target_train.csv',index_col=0)
target_test = pd.read_csv('../Data/target_test.csv',index_col=0)
target = pd.concat((target_train,target_test))

features_train = pd.read_csv('../Data/features_train_scaled.csv')
features_test = pd.read_csv('../Data/features_test_scaled.csv')
features = pd.concat((features_train,features_test))

app.layout = html.Div([

    html.Div(children=[
        dcc.Markdown('''
                Redox flow batteries are a promising class of electrochemical energy storage devices. The difference in voltages between the active molecules on the 'reduction' and 'oxidation' 
                sides determines the operational voltage of the battery. This dashboard lists 5442 molecules, whose reduction potentials are predicted using ML models. 
                The code repository including data, model pipelines is available [here](https://github.com/adisun94/DataScience) and [here](https://github.com/akashjn/DataScience). 
                More information about Redox Flow Batteries is available [here](https://energystorage.org/why-energy-storage/technologies/redox-flow-batteries/)''')
        ],
        style={'padding': 10, 'flex': 1, 'background-color' : 'rgb(255,255,240)'}),
    html.Br(),
    html.Div(children=[
        html.Label(['Select colorbar property'], style={'font-weight': 'bold', "text-align": "left"}),
        dcc.Dropdown(
            df.columns.drop(['ERed','MolLogP','SMILES']),
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
        ),
        dcc.Graph(
            id='predictions',
            figure={}
        )],
        style={'display': 'flex', 'flex-direction': 'row', 'background-color' : 'rgb(255,255,240)'})
    ])
    
@app.callback(
    Output('scatterplot','figure'),
    Input('property','value')
)

def scatter_plot(value):

    if value=='MolWT':
        units='g/mol'
    if value=='NValE':
        units=''

    fig = go.Figure(data=[
    go.Scatter(
        x=df["MolLogP"],
        y=df["ERed"],
        mode="markers",
        marker=dict(
            colorscale='viridis',
            color=df[value],
            size=df["MolWT"],
            colorbar={"title": value+' ('+units+')'},
            line={"color": "#444"},
            reversescale=True,
            sizeref=45,
            sizemode="diameter",
            opacity=0.8,
            )
        )])
       
    fig.update_layout(width=800,height=650)

    fig.update_traces(hoverinfo="none", hovertemplate=None)

    fig.update_layout(
        #title='Molecule dataset : Redoxmer electrolytes for Redox Flow Batteries (RFB)',
        xaxis=dict(title='MolLogP'),
        yaxis=dict(title='ERed (V)'),
        font=dict(family="Arial",size=18),
        plot_bgcolor='rgba(220,220,220,1)',
        paper_bgcolor='rgba(255,255,240,1)'
        )

    return fig

@app.callback(
    Output('moleculeimage','figure'),
    Output('predictions','figure'),
    Input('scatterplot','hoverData'),
    Input('property','value')
)

def molecule_image(hoverData,value):

    if value=='MolWT':
        units='g/mol'
    if value=='NValE':
        units=''

    if hoverData is None:

        potential = {'method': ['DFT','LinearReg','GBR','NN'], 'potential': [0,0,0,0]}
        potential=pd.DataFrame.from_dict(potential)

        img = io.imread('assets/molecule.jpg')
        fig = px.imshow(img)

        fig.update_layout(width=500,height=500)

        fig.update_traces(hoverinfo="none", hovertemplate=None)

        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False),

        fig.update_layout(
            plot_bgcolor='rgba(255,255,240,1)',
            paper_bgcolor='rgba(255,255,240,1)',
            margin=dict(l=100, r=20, t=150, b=00)
            )

        fig2 = px.bar(potential,
        x='method',y='potential')
        
       
        fig2.update_layout(width=400,height=600)

        fig2.update_layout(
        #title='Molecule dataset : Redoxmer electrolytes for Redox Flow Batteries (RFB)',
        xaxis=dict(title='Method'),
        yaxis=dict(title='ERed (V)'),
        font=dict(family="Arial",size=18),
        plot_bgcolor='rgba(220,220,220,1)',
        paper_bgcolor='rgba(255,255,240,1)',
        margin=dict(l=100, r=20, t=100, b=00)
        )

        return fig, fig2

    pt = hoverData["points"][0]
    num = pt["pointNumber"]

    df_row = df.iloc[num]
    img_src=io.imread('assets/molID_'+str(num)+'.png')

    dft_pot=float(target[target.SMILES==df.SMILES[num]]['ERed'])
    feat_num=features[target.SMILES==df.SMILES[num]]

    lr_pred=lr_model.predict(np.array(feat_num).reshape(1,-1))
    gbr_pred=gbr_model.predict(np.array(feat_num).reshape(1,-1))
    nn_pred=nn_model.predict(np.array(feat_num).reshape(1,-1))

    potential={'method':['DFT','LinearReg','GBR','NN'],
               'potential':[dft_pot,lr_pred[0],gbr_pred[0],nn_pred[0][0]]}
    potential=pd.DataFrame.from_dict(potential)

    fig = px.imshow(img_src)

    fig.update_layout(width=500,height=500)

    fig.update_traces(hoverinfo="none", hovertemplate=None)

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False),

    fig.update_layout(
        title='<br>Molecule = '+df.SMILES[num]+'<br>'+str(value)+' = '+str(df[value][num])+' '+units,
        font=dict(family="Arial",size=18),
        plot_bgcolor='rgba(255,255,240,1)',
        paper_bgcolor='rgba(255,255,240,1)',
        margin=dict(l=100, r=20, t=100, b=0)
        )

    
    fig.update_layout(
        plot_bgcolor='rgba(255,255,240,1)',
        paper_bgcolor='rgba(255,255,240,1)'
        )

    fig2 = px.bar(potential,
    x='method',y='potential')
    
    
    fig2.update_layout(width=400,height=600)

    fig2.update_layout(
    #title='Molecule dataset : Redoxmer electrolytes for Redox Flow Batteries (RFB)',
    xaxis=dict(title='Method'),
    yaxis=dict(title='ERed (V)'),
    font=dict(family="Arial",size=18),
    plot_bgcolor='rgba(220,220,220,1)',
    paper_bgcolor='rgba(255,255,240,1)',
    margin=dict(l=100, r=20, t=100, b=00)
    )

    return fig, fig2

if __name__ == "__main__":
    app.run_server(debug=True)