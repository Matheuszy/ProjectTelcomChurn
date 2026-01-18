from fastapi import FastAPI
from pydantic import BaseModel
from src.ChurnPredictor import ChurnPredictor
import pandas as pd


class ClienteData(BaseModel):
    CidadaoSenior: int
    Parceiro: int
    Dependentes: int
    Fidelidade: int
    ServicoTelefonico: int
    SuporteTecnico: int
    StreamingTV: int
    MensalCobrado: float
    TotalCobrado: float
    MultiplasLinhas_No: int
    MultiplasLinhas_No_phone_service: int
    MultiplasLinhas_Yes: int
    ServicoInternet_DSL: int
    ServicoInternet_Fiber_optic: int
    ServicoInternet_No: int
    Contrato_Month_to_month: int
    Contrato_One_year: int
    Contrato_Two_year: int

app = FastAPI()
previsor = ChurnPredictor("./model/modelo_churn_final.pkl")

@app.post("/prever")
def prever_churn(cliente: ClienteData):

    colunas_modelo = [
        "CidadaoSenior", "Parceiro", "Dependentes", "Fidelidade",
        "ServicoTelefonico", "SuporteTecnico", "StreamingTV",
        "MensalCobrado", "TotalCobrado", "MultiplasLinhas_No",
        "MultiplasLinhas_No phone service",
        "MultiplasLinhas_Yes",
        "ServicoInternet_DSL",
        "ServicoInternet_Fiber optic",
        "ServicoInternet_No",
        "Contrato_Month-to-month",
        "Contrato_One year",
        "Contrato_Two year"
    ]


    dados = cliente.model_dump()


    mapeamento = {
        "MultiplasLinhas_No_phone_service": "MultiplasLinhas_No phone service",
        "ServicoInternet_Fiber_optic": "ServicoInternet_Fiber optic",
        "Contrato_Month_to_month": "Contrato_Month-to-month",
        "Contrato_One_year": "Contrato_One year",
        "Contrato_Two_year": "Contrato_Two year"
    }


    dados_corrigidos = {mapeamento.get(k, k): v for k, v in dados.items()}


    df_cliente = pd.DataFrame([dados_corrigidos])[colunas_modelo]

    classe, proba = previsor.predict_customer(df_cliente)

    return {
        "status_churn": "Sim" if classe[0] == 1 else "Não",
        "probabilidade": f"{proba[0][1]:.2%}"
    }
