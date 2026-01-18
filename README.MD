# 📡 Predição de Churn em Telecomunicações com Autenticação Java

Este projeto é uma solução completa que utiliza **Machine Learning** para prever o cancelamento de clientes (Churn) e um **Sistema de Autenticação** robusto para gerenciar o acesso às previsões.

O repositório está organizado em uma arquitetura de microsserviços simples (Monorepo):
1.  **`python_service`**: API de Inteligência Artificial.
2.  **`java_auth_system`**: Cliente de autenticação e interface de usuário.

---

## 🚀 Como Executar o Projeto

Para o funcionamento completo, você precisará rodar os dois serviços simultaneamente.

### 1. Backend de Inteligência Artificial (Python)
Este serviço contém o modelo treinado e a API que processa os dados.

* **Tecnologias:** Python 3.10+, FastAPI, Scikit-Learn, Pandas.
* **Como rodar:**
    ```bash
    cd python_service
    pip install -r requirements.txt
    python main.py  # Ou o comando que você usa para iniciar a API
    ```
    *A API ficará disponível em: `http://localhost:8000`*

### 2. Sistema de Autenticação e Cliente (Java)
Este serviço gerencia o login do usuário e envia os dados para a API Python.

* **Tecnologias:** Java 17+, Spring Boot, Maven.
* **Como rodar:**
    ```bash
    cd java_auth_system
    ./mvnw spring-boot:run
    ```
    *O sistema de login estará disponível em: `http://localhost:8080`*

---

## 🛠️ Arquitetura de Comunicação

O fluxo de dados do projeto funciona da seguinte forma:
1.  O usuário faz login através do **Java Auth System**.
2.  Após autenticado, o usuário insere os dados do cliente de telecomunicações.
3.  O código Java realiza uma requisição `POST` para o **Python Service**.
4.  O modelo de ML processa os dados e retorna a probabilidade de Churn.
5.  O Java exibe o resultado final para o usuário.

---

## 📂 Estrutura de Pastas

```text
ProjectTelcomChurn/
├── python_service/      # API, Modelos (.pkl) e scripts de treino.
├── java_auth_system/    # Código-fonte Spring Boot e lógica de segurança.
└── README.md            # Documentação principal.
