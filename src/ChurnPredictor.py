import joblib
import pandas as pd

class ChurnPredictor:
    def __init__(self, model_path):
        # Carrega o modelo salvo no disco ao iniciar a classe
        self.model = joblib.load(model_path)
        print(f"Modelo carregado de: {model_path}")

    def predict_customer(self, customer_data_df):
        """
        Recebe um DataFrame com 1 ou mais clientes e retorna
        a previsão e a probabilidade.
        """
        previsao = self.model.predict(customer_data_df)
        probabilidade = self.model.predict_proba(customer_data_df)

        return previsao, probabilidade