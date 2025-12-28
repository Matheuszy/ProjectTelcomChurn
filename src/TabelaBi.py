import pandas as pd


class TabelaBi:
    def __init__(self):
        importancia = pd.DataFrame({
            'Atributo': x_train.columns,
            'Peso': clf.feature_importances

        })
        return importancia.to_csv('importancia_features.csv', index=False)
