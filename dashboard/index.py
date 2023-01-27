from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Small molecule drugbank dataset
data_path = 'dashboard_data_5442mols.csv'

df = pd.read_csv(data_path, header=0, index_col=0)

fig = go.Figure(data=[
    go.Scatter(
        x=df["MolLogP"],
        y=df["ERed"],
        mode="markers",
        marker=dict(
            colorscale='viridis',
            color=df["MolWT"],
            size=df["MolWT"],
            colorbar={"title": "Molecular<br>Weight (g/mol)"},
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
       plot_bgcolor='rgba(246,252,211,1)',
       paper_bgcolor='rgba(220,220,220,1)'
        )

app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
    dcc.Tooltip(id="graph-tooltip"),
    dcc.Markdown('''
                #### This dashboard lists 5442 molecules, whose reduction potentials are predicted using a Gradient Boosting Regression model. The code repository is available [here](https://github.com/adisun94/DataScience) and [here](https://github.com/akashjn/DataScience). More information about RFB available [here](https://energystorage.org/why-energy-storage/technologies/redox-flow-batteries/)''')
    ])


@app.callback(
        Output("graph-tooltip", "show"),
       Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Input("graph-basic-2", "hoverData"),
        )
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]

    df_row = df.iloc[num]
    img_src=r'assets/molID_'+str(num)+'.png'
    redpot = df_row['ERed']
    desc = df.index[num]

    children = [
            html.Div([
                html.Img(src=img_src, style={"width": "100%"}),
                html.P(f"Reduction Potential = {redpot} V"),
                html.P(f"{desc}"),
                ], style={'width': '250px', 'white-space': 'normal'})
                                                                                            ]

    return True, bbox, children


if __name__ == "__main__":
    app.run_server(debug=True)

