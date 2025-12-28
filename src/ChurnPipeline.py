import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib


class ChurnPipeline:
    def __init__(self, depth=5):
        self.model = DecisionTreeClassifier(max_depth=depth, class_weight="balanced")


    def preprocess_data(self, df):
        df_limpo = pd.get_dummies(df, columns=['MultiplasLinhas', 'ServicoInternet', 'Contrato'])
        return df_limpo

    def train(self, x, y):
        self.model.fit(x,y)
        print("Modelo treinado com sucesso")


    def save_model(self, filename):

        joblib.dump(self.model, filename)


