import pandas as pd
from ChurnPipeline import ChurnPipeline
from ModeloDecisionTree import ModeloDecisionTree
from TabelaBi import TabelaBi
from sklearn.model_selection import train_test_split
from ChurnPredictor import ChurnPredictor

# Carregar dados
try:
    df = pd.read_csv("database.csv")
except FileNotFoundError:
    print("Erro: O arquivo 'database.csv' não foi encontrado na pasta do projeto!")
    exit()

# Instanciar classes
pipe = ChurnPipeline()
tree_model = ModeloDecisionTree()
bi = TabelaBi()

# Processar dados

df_limpo = pipe.preprocess_data(df)


X = df_limpo.drop('Churn', axis=1, errors='ignore')
y = df['Churn'].replace({'No': 0, 'Yes': 1})

# Dividir Treino e Teste
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Executar
pipe.train(x_train, y_train)
bi.gerando_tabela(pipe.model, x_train.columns)
tree_model.gerando_resultado_final(pipe.model, x_test)
pipe.save_model("modelo_churn_final.pkl")
print("Modelo persistido com sucesso como 'modelo_churn_final.pkl'!")
previsor = ChurnPredictor("modelo_churn_final.pkl")

novo_cliente = pd.DataFrame([{
    'CidadaoSenior': 0, 'Parceiro': 1, 'Dependentes': 0, 'Fidelidade': 1,
    'ServicoTelefonico': 1, 'SuporteTecnico': 0, 'StreamingTV': 0,
    'MensalCobrado': 70.0, 'TotalCobrado': 70.0,
    'MultiplasLinhas_No': 1, 'MultiplasLinhas_No phone service': 0, 'MultiplasLinhas_Yes': 0,
    'ServicoInternet_DSL': 1, 'ServicoInternet_Fiber optic': 0, 'ServicoInternet_No': 0,
    'Contrato_Month-to-month': 1, 'Contrato_One year': 0, 'Contrato_Two year': 0
}])


classe, proba = previsor.predict_customer(novo_cliente)

print(f"Resultado: {'Churn' if classe[0] == 1 else 'Fica'}")
print(f"Risco: {proba[0][1]:.2%}")