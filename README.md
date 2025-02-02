# Generative VAE Latent Analysis 

Este projeto explora a formação do espaço latente usando **Variational Autoencoders (VAE)** em dois conjuntos de dados com volumes diferentes, permitindo uma análise comparativa.

##  Datasets Utilizados
1️⃣ **UCI Covertype (Forest Cover Type)**  
   -  ~581.000 amostras |  54 features  
   -  Classificação de tipos de cobertura do solo.  
   -  Arquivo: `data/forest_cover.csv`  

2️⃣ **Wine Dataset**  
   -  178 amostras | 🔹13 features  
   -  Classificação de tipos de vinho com base em propriedades químicas.  
   -  Arquivo: `data/wine.csv`  

## Objetivos do Projeto
- Treinar **Variational Autoencoders (VAE)** para gerar representações latentes dos datasets.
- Projetar o **espaço latente em 2D** usando **PCA** e analisar a separação das classes.
- Comparar a **formação do espaço latente** entre um dataset pequeno e um grande.

## Estrutura do Projeto
 `src/download_dataset.py` → Script para baixar o UCI Covertype.  
 `src/download_wine.py` → Script para baixar o Wine Dataset.  
 `src/exploracao.py` → Análise exploratória dos dados.  
 `src/preprocess.py` → Pré-processamento dos dados (normalização).  
 `src/train_vae.py` → Implementação e treinamento do modelo VAE.  
 `data/` → Diretório com os datasets.  
 `data/processed/` → Diretório com os datasets normalizados.  
 `models/` → Diretório para salvar modelos treinados.  

##  Como Rodar o Projeto?
1️⃣ **Criar ambiente virtual e instalar dependências**  
```powershell
python -m venv venv
venv\Scripts\Activate
pip install -r requirements.txt
