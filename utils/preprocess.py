import pandas as pd
import numpy as np


class DataProcessor:
    def __init__(self ):
        self.df_large = None
        self.df_small = None



    def sort_data(self, data):
        """
            Create a Function to sort the dataframe based on the timestamp column before extracting the cumulative
            returns the dataframe with three columns sorted

        """
        data["TimeStamp"]  =pd.to_datetime(data['TimeStamp'])
        data = data.sort_values("TimeStamp")
        data.drop("TimeStamp" , inplace = True , axis=1)
        
        return data


    def extract_data(self ,df):
        df_copy = df.copy()
        
        df_copy['RunLength'] = df_copy["MemoField"].str.extract(r'RunLength=[^\d]*(\d+(?:\.\d+)?)');
        df_copy["N_Pulses"] =  df_copy["MemoField"].str.extract(r'NumberOfPulses=[^\d]*(\d+(?:\.\d+)?)');
        df_copy["NewCFactor"] = df_copy["MemoField"].str.extract(r'NewCFactor=[^\d]*(\d+(?:\.\d+)?)');
        df_copy = df_copy[["TimeStamp" ,"NewCFactor" , "N_Pulses" , "RunLength" ]]   

        for col in ["N_Pulses","NewCFactor" , "RunLength"]:
            df_copy[col] = df_copy[col].astype("float64")

        return df_copy


    def add_cum(self ,df):
        df_copy = df.copy()
        df_copy["RunLength_Cum"] = np.cumsum(df_copy["RunLength"])
        df_copy["N_Pulses_Cum"] = np.cumsum(df_copy["N_Pulses"])

        return df_copy


    def prep_data(self,  df = None):
        if df is None:
            raise Exception("No data were passed , pass dataframe")
        elif isinstance(self.df_large , pd.DataFrame ) and isinstance(self.df_small , pd.DataFrame):
            return self.df_large , self.df_small
            
        pd.set_option('display.precisdion', 15)

        df_copy = df.copy()



        print(f"Extracting Data from Message Column.......")
        df_large = df[df.apply(lambda row: row.astype(str).str.contains('\[Focus=Largest\]').any(), axis=1)];
        df_small = df_copy[df_copy.apply(lambda row: row.astype(str).str.contains('\[Focus=Smallest\]').any(), axis=1)];
        
        
        ## Extract data

        self.df_large = self.extract_data(df_large)  
        
        self.df_large = self.sort_data(self.df_large)


        self.df_large = self.add_cum(self.df_large)
        
        
        ############# smallest

        self.df_small = self.extract_data(df_small)
        
        self.df_small = self.sort_data(self.df_small)


        self.df_small = self.add_cum(self.df_small)

        

        
        
        
        return self.df_large , self.df_small


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df = pd.read_xml("data/KUH.XML")
    processor = DataProcessor()
    df_large ,df_small = processor.prep_data(df)

    plt.plot(df_large.NewCFactor.values)
    plt.show()

    




