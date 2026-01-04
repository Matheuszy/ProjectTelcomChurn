import pandas as pd
from ChurnPipeline import ChurnPipeline
from ModeloDecisionTree import ModeloDecisionTree
from TabelaBi import TabelaBi
from sklearn.model_selection import train_test_split
from ChurnPredictor import ChurnPredictor
from sklearn.model_selection import cross_val_score


# Carregar dados
try:
    df = pd.read_csv("../data/database.csv")
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
y = df['Churn'].replace({'No': 0, 'Yes': 1}).infer_objects(copy=False)
y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)

# Dividir Treino e Teste
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Executar
pipe.train(x_train, y_train)
bi.gerando_tabela(pipe.model, x_train.columns)
print("\n--- Iniciando Validação Cruzada ---")
# usando validação cruzada
scores = cross_val_score(pipe.model, X, y, cv=5)

print(f"Acurácias em cada fatia: {scores}")
print(f"Média de Acurácia: {scores.mean():.2%}")
print(f"Desvio Padrão: {scores.std():.2%}")

if scores.std() < 0.05:
    print("✅ O modelo é estável! A variação entre os testes é baixa.")
else:
    print("⚠️ Atenção: O modelo está variando muito dependendo dos dados.")
tree_model.gerando_resultado_final(pipe.model, x_test)
pipe.save_model("../model/modelo_churn_final.pkl")
print("Modelo persistido com sucesso como 'modelo_churn_final.pkl'!")
previsor = ChurnPredictor("../model/modelo_churn_final.pkl")

novo_cliente = pd.DataFrame([{
    'CidadaoSenior': 1, 'Parceiro': 1, 'Dependentes': 0, 'Fidelidade': 1,
    'ServicoTelefonico': 0, 'SuporteTecnico': 0, 'StreamingTV': 0,
    'MensalCobrado': 120.0, 'TotalCobrado': 90.0,
    'MultiplasLinhas_No': 1, 'MultiplasLinhas_No phone service': 0, 'MultiplasLinhas_Yes': 0,
    'ServicoInternet_DSL': 1, 'ServicoInternet_Fiber optic': 1, 'ServicoInternet_No': 0,
    'Contrato_Month-to-month': 1, 'Contrato_One year': 1, 'Contrato_Two year': 0
}])


classe, proba = previsor.predict_customer(novo_cliente)

print(f"Resultado: {'Churn' if classe[0] == 1 else 'Fica'}")
print(f"Risco: {proba[0][1]:.2%}")