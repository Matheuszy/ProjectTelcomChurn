📊 Predição de Churn em Telecomunicações
Este projeto utiliza Machine Learning para identificar clientes com alta probabilidade de cancelar serviços de telecomunicações (Churn). A solução foi estruturada seguindo boas práticas de engenharia de software, utilizando Programação Orientada a Objetos (POO) e uma pipeline automatizada para facilitar a integração com ferramentas de Business Intelligence como o Power BI.

🛠️ Tecnologias Utilizadas
Python 3.10+

Pandas & Numpy: Manipulação e tratamento de dados.

Scikit-Learn: Criação do modelo de Árvore de Decisão.

Joblib: Persistência do modelo treinado (formato .pkl).

Power BI: Visualização estratégica dos resultados.

🏗️ Estrutura do Projeto
O código está dividido em módulos para garantir escalabilidade e fácil manutenção:

ChurnPipeline.py: Contém a lógica de limpeza de dados, engenharia de atributos (Dummies, Rename, Replace) e o motor de treinamento.

ModeloDecisionTree.py: Gera a base de dados final com as predições para análise em massa.

TabelaBi.py: Extrai os pesos das variáveis (Feature Importance) para análise de causa raiz.

ChurnPredictor.py: Classe responsável por carregar o modelo salvo e realizar predições em tempo real para novos clientes.

main.py: O orquestrador que executa o fluxo completo do projeto.

🚀 Como Executar
Certifique-se de ter as bibliotecas instaladas:



pip install -r requirements.txt
Coloque o arquivo database.csv na raiz do projeto.

Execute o arquivo principal:



python src/main.py
📈 Resultados e Integração com BI
O sistema gera dois arquivos CSV automáticos:

importancia_features.csv: Mostra quais fatores (como tipo de contrato ou fidelidade) mais influenciam a saída do cliente.

base_com_previsoes.csv: Lista detalhada de clientes com a "flag" de risco, pronta para ser usada por times de retenção.


Shutterstock
🧠 Lógica do Modelo
Foi utilizado o algoritmo Decision Tree Classifier com ajuste de class_weight='balanced'. Esta escolha foi feita para priorizar o Recall (identificar o máximo de clientes em risco), garantindo que a empresa possa agir preventivamente, mesmo em cenários onde a acurácia global possa variar.