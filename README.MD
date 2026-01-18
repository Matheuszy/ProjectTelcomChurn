# 📡 Telecom Churn Prediction & Authentication System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Java](https://img.shields.io/badge/Java-17%2B-orange?logo=openjdk)](https://www.oracle.com/java/)
[![Spring Boot](https://img.shields.io/badge/Spring_Boot-3.x-green?logo=springboot)](https://spring.io/projects/spring-boot)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688?logo=fastapi)](https://fastapi.tiangolo.com/)

## 📝 Descrição do Projeto
Esta solução integrada combina **Data Science** e **Desenvolvimento Backend** para resolver o problema de retenção de clientes em telecomunicações. O ecossistema permite que usuários autenticados via Java submetam dados de consumo para obter predições em tempo real processadas por um modelo de Machine Learning em Python.

---

## 🏗️ Arquitetura do Sistema (Monorepo)

O repositório utiliza uma estrutura de microsserviços desacoplados para garantir escalabilidade e organização:

* **`python_service/`**: Core de Inteligência Artificial. Contém o modelo treinado (`.pkl`), scripts de pré-processamento e a API de inferência.
* **`java_auth_system/`**: Sistema de gestão de acesso desenvolvido com Spring Boot. Responsável pela autenticação de usuários e por atuar como cliente que consome o serviço de predição.

---

## 🛠️ Tecnologias Utilizadas

### Inteligência Artificial & API (Python)
* **Pandas & NumPy**: Manipulação e tratamento de dados.
* **Scikit-Learn**: Algoritmo de Árvore de Decisão para classificação.
* **Joblib**: Persistência do modelo de ML.
* **FastAPI**: Framework para exposição do modelo via REST.

### Autenticação & Integração (Java)
* **Spring Boot 3**: Framework base para a aplicação Java.
* **Spring Security**: Controle de acesso e proteção de rotas.
* **Spring Data JPA**: Integração com banco de dados para usuários.
* **RestTemplate/WebClient**: Comunicação síncrona com a API Python.

---

## 🚀 Guia de Instalação e Execução

### 1. Backend de Machine Learning (Python)
```bash
cd python_service
python -m venv venv
# No Windows:
venv\Scripts\activate  
# No Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt
python main.py


Disponível em: http://localhost:8000

cd java_auth_system
./mvnw clean install
./mvnw spring-boot:run

http://localhost:8080

Fluxo de Comunicação
Origem (Client),Destino (API),Método,Função
Java Auth System,Python ML Service,POST,Envio de dados de consumo para predição.
Browser/Frontend,Java Auth System,POST,Autenticação e login do operador.



ProjectTelcomChurn/
├── python_service/      # API, Modelos (.pkl) e scripts de treino Python
├── java_auth_system/    # Sistema de Autenticação e Cliente Spring Boot
└── README.md            # Documentação Geral do Projeto
