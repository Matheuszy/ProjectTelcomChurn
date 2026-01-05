import pandas as pd

class ModeloDecisionTree:
    def __init__(self):
        pass

    def gerando_resultado_final(self, model, x_test):
        # Cria a base com as previsões (0 ou 1)
        resultado_final = x_test.copy()
        resultado_final['Previsao_Churn'] = model.predict(x_test)


        resultado_final.to_csv('../model/base_com_previsoes.csv', index=False)
        print("Arquivo base_com_previsoes.csv gerado!")