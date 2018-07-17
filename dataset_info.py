import pandas as pd
import matplotlib.pyplot as plt

class DatasetInfo:
    df = []

    def setup(self, data_dict):
        self.df = pd.DataFrame.from_dict(data_dict, orient = 'index')

    def print_dataframe_info(self):
        # Transform data from dictionary to the Pandas DataFrame
        self.df.info()
        print(self.df.describe().transpose())

    def plot_outlier(self):
        salary_bonus = self.df[['salary', 'bonus']].astype(float)
        salary_bonus.plot('salary', 'bonus', 'scatter')
        plt.show()