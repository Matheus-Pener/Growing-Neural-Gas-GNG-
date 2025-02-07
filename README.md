#  Análise de Agrupamento com Growing Neural Gas (GNG)

Este projeto investiga **modelos neurais não supervisionados** aplicados ao **Wine Dataset**, utilizando a técnica **Growing Neural Gas (GNG)** para mapear padrões.

---

##  **Dataset Utilizado**
 **Wine Dataset**  
   - 178 amostras | 🔹13 features  
   - Classificação de tipos de vinho baseada em propriedades químicas.  
   - Arquivo: `data/wine.csv`  

---

## 🎯 **Objetivos do Projeto**
✅ **Treinar um modelo GNG** para detectar padrões e agrupamentos no Wine Dataset.  
✅ **Avaliar a formação dos clusters** e identificar possíveis **outliers**.  
✅ **Explorar variações nos parâmetros** (`learning_rate`, `max_age`, `número de neurônios`, seleção de features).  
✅ **Comparar a segmentação ao utilizar todas as variáveis vs. um subconjunto otimizado.**  

---

## **Metodologia**
1️⃣ **Exploração inicial dos dados** (distribuição das features, matriz de correlação).  
2️⃣ **Normalização dos dados** para garantir comparabilidade.  
3️⃣ **Treinamento do GNG** utilizando diferentes configurações.  
4️⃣ **Análise dos gráficos gerados** para interpretar a segmentação dos clusters.  

---

## 📊 **Resultados e Análises**

### **📌 Utilização das 5 dimensões mais correlacionadas para melhorar a segmentação**
📌 **Correlação entre os dados do Wine Dataset:**  
![Correlação entre os dados](image-1.png)

### 🔹 **Clusters e Outliers**
- O modelo **GNG detectou padrões latentes**, mostrando como os dados se organizam naturalmente.  
- Os clusters apresentaram **uma organização linear**, indicando forte correlação entre algumas variáveis.  
- Algumas conexões mais longas indicam regiões menos povoadas do espaço latente.

### 🔹 **Homogeneidade dos Agrupamentos**
- Os neurônios do GNG se ajustaram a **regiões de maior densidade**, formando agrupamentos coerentes.  
- **Reduzir as features** para as 5 mais correlacionadas melhorou a segmentação e destacou um padrão linear.

### 🔹 **Variação de Parâmetros**
- **Ajustes de `max_age` quebraram conexões entre grupos, aumentando a separação.**  
- **Ajustar `learning_rate` permitiu que os neurônios se acomodassem mais rápido aos padrões dos dados.**  
- **Escolher apenas as 5 variáveis mais conectadas melhorou a clareza dos clusters.**  

### 🔹 **Gráfico gerado no Wine Dataset com as 5 variáveis mais correlacionadas**
📌 **Clusters identificados pelo GNG:**  
![Clusters formados pelo GNG](image.png)  