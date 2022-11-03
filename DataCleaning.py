import pandas as pd

class patch_year:
        
    start=2000
    start_year=0
    end_year=0
    end=2022
    cum_sales={}
    sales=pd.DataFrame()

    def __init__(self,state,data):
        self.state=state
        self.data=data
                                                    
    def starting_year(self):
        self.start_year=min(self.data['Date'].groupby(pd.to_datetime(self.data['Date']).dt.year).count().index)
        return self.start_year
                                                                    
    def ending_year(self):
        self.end_year=max(self.data['Date'].groupby(pd.to_datetime(self.data['Date']).dt.year).count().index)
        return self.end_year
                                                                                
    def year_vs_make(self):
        return pd.crosstab(pd.to_datetime(self.data['Date']).dt.year,self.data['Make'])
                                                                                            
    def total_sales(self):
        cum_sum=self.data.groupby(pd.to_datetime(self.data['Date']).dt.year).count()['Date'].cumsum()
        year_data=cum_sum.index
        self.cum_sales=cum_sum
        self.cum_sales[self.start]=0
        for y in range(self.start+1,self.end+1,1):
            if y not in year_data:
                self.cum_sales[y]=self.cum_sales[y-1]
        self.cum_sales.sort_index(inplace=True)
        self.cum_sales=self.cum_sales.to_frame()
        self.cum_sales.rename({'Date':'Count'},axis=1,inplace=True)
        self.cum_sales.reset_index(inplace=True)
        return self.cum_sales

    def five_most_makes(self):
        self.five_most_sales=[]
        for y in range(self.year_vs_make().shape[0]):
            self.five_most_sales.append(self.year_vs_make().iloc[y].nlargest(5).to_frame().T)
        return self.five_most_sales
