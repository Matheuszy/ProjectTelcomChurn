import pandas as pd

class TabelaBi:
    def __init__(self):
        pass

    def gerando_tabela(self, model, feature_columns):
        # Extrai a importância que o modelo deu para cada coluna
        importancia = pd.DataFrame({
            'Atributo': feature_columns,
            'Peso': model.feature_importances_
        })


        importancia = importancia.sort_values(by='Peso', ascending=False)
        importancia.to_csv('../model/importancia_features.csv', index=False)
        print("Arquivo importancia_features.csv gerado!")