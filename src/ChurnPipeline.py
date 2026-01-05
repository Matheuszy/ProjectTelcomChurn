import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

class ChurnPipeline:
    def __init__(self, depth=5):
        pd.set_option('future.no_silent_downcasting', True)
        self.model = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=42
        )

    def preprocess_data(self, df):
        # Remoção de colunas
        colunas_removidas = ["customerID", "gender", "StreamingMovies", "PaperlessBilling",
                             "OnlineBackup", "OnlineSecurity", "DeviceProtection", "PaymentMethod"]
        df = df.drop(colunas_removidas, axis=1, errors='ignore')

        # Renomear colunas
        df = df.rename(columns={
            "SeniorCitizen": "CidadaoSenior",
            "Partner": "Parceiro",
            "Dependents": "Dependentes",
            "tenure": "Fidelidade",
            "PhoneService": "ServicoTelefonico",
            "MultipleLines": "MultiplasLinhas",
            "InternetService": "ServicoInternet",
            "TechSupport": "SuporteTecnico",
            "StreamingTV": "StreamingTV",
            "Contract": "Contrato",
            "MonthlyCharges": "MensalCobrado",
            "TotalCharges": "TotalCobrado"
        })


        colunas_binarias = ["Parceiro", "Dependentes", "ServicoTelefonico", "SuporteTecnico", "StreamingTV"]
        df[colunas_binarias] = df[colunas_binarias].replace({
            'No': 0, 'Yes': 1, 'No internet service': 0, 'No phone service': 0
        })

        # Conversão de tipos
        df['TotalCobrado'] = pd.to_numeric(df['TotalCobrado'], errors='coerce').fillna(0)

        # Criação de Dummies
        df_final = pd.get_dummies(df, columns=['MultiplasLinhas', 'ServicoInternet', 'Contrato'])

        return df_final

    def train(self, x, y):
        self.model.fit(x, y)
        print("Modelo treinado com sucesso!")

    def save_model(self, path):
        joblib.dump(self.model, path)
        print(f"Modelo persistido com sucesso como '{path}'!")