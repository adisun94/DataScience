import numpy as np
import pandas as pd
from DataCleaning import *

class bp:

    data_hist=np.empty([0,3])
    data=pd.DataFrame()

    def __init__(self,state):
        self.state=state

    def top_five(self):
        self.data=pd.read_json('../dataUSA/USA/'+self.state+'_EVdata.json',orient='records')
        if 'Date' in (self.data).columns:
            p=patch_year(self.state,self.data)
            t=p.five_most_makes()
        for i in t:
            for j in i.columns:
                self.data_hist=np.vstack((self.data_hist,(i.index[0],j,int(i[str(j)]))))

        self.data_hist=pd.DataFrame(self.data_hist, columns=['Year','Make','Count'])
        self.data_hist['Year']=self.data_hist['Year'].astype(int)
        self.data_hist['Count']=self.data_hist['Count'].astype(int)

        return self.data_hist


