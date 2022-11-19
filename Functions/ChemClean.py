import pandas as pd
import numpy as np

class clean:

    """Class to perform data cleaning operations 

    .. attribute:: df
      
       Input data frame whose columns are 208 descriptors generated from the RDkit package

    """

    def __init__(self,data):
        self.df=data

    def replace_nan(self):
        """
        Method to replace NaN cells with average value of that feature (computed using other numerical entries)

        Returns:
        DataFrame with reduced features

        """

        self.df.fillna(value=self.df.mean(), inplace=True)
        
        return self.df

    def remove_nan(self):
        """
        Method to eliminate features which contain a NaN entry in any data entry.

        Returns:
        DataFrame with reduced features

        """
        
        # check columns with nan
        columns_with_nan=self.df.columns[self.df.isna().any()]
        print(f"Removing {len(columns_with_nan)} columns with nan, if any")
        print(f"Before: df.shape={self.df.shape}")
        self.df = self.df.dropna(axis='columns')
        # df = df.drop(columns=columns_with_nan)
        print(f"After: df.shape={self.df.shape} \n")    
                                    
        return self.df
    
    def remove_duplicate(self):
        """
        Method to eliminate duplicated features.

        Returns:
        DataFrame with reduced features

        """
        
        # Remove duplicates
        print(f"removing {sum(df.columns.duplicated())} duplicate columns, if any")
        print(f"Before: df.shape={self.df.shape}")
        self.df=self.df.loc[:,~self.df.columns.duplicated()].copy()
        print(f"After: df.shape={self.df.shape} \n")

        return self.df

    def remove_unique(self):    
        """
        Method to eliminate features which have a unique value for all data entries

        Returns:
        DataFrame with reduced features

        """
        
        # Remove columns with a unique value; will also remove columns with only 0 
        print(f"removing {sum(self.df.nunique()<2)} columns values with a unique value")
        print(f"Before: df.shape={self.df.shape}")
        self.df=self.df.loc[:,self.df.nunique()>1]
        print(f"After: df.shape={self.df.shape} \n")

        return self.df

    def remove_columns_low_std(self,t=0.3):
        """
        Method to eliminate features whose standard deviation is less than a threshold t, set to a default value of 0.3.
        
        Args:
        t: Standard deviation of feature with a default value of 0.3

        Returns:
        DataFrame with reduced features
        
        """
       
        self.threshold=t
        print(f"Removed {sum(self.df.std() < self.threshold)} columns with std < {self.threshold} ")
        self.df=self.df.loc[:, self.df.std() >= self.threshold]
        return self.df

    def remove_corr_features(self,c=0.5):
        """
        Method to eliminate highly correlated featurest.

        Args:
        c: Cutoff value for the Pearson correlation coefficient with a default value of 0.5.

        Returns:
        DataFrame with reduced features
       
        """
       
        self.corr_cutoff=c
        cor_matrix=self.df.corr().abs()
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool_))

        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.corr_cutoff)]
        print(f"Dropped {len(to_drop)} features with correlation coeff. > {self.corr_cutoff:0.2f}")

        self.df=self.df.drop(columns=to_drop,axis=1)
        return self.df
