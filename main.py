# Instalar Dependências
#
# pip install pandas
# pip install matplotlib
# pip install scikit-learn
# pip install seaborn
# pip install tensorflow

# Módulos do Usuário
from config import split_year, epochs, batch_size
from user_text_lib import p, secao, list_station
from user_data_lib import prediction, data_profile, analyse_pax
from user_graph_lib import plot_line, plot_minmax, plot_boxplot, plot_correlation_heatmap, plot_scatter

# Outros Módulos -- Instalar Dependências
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

### Importacao e Analise de Dados do Metro

# Carregar o arquivo CSV de dados do metrô
metro_data = pd.read_csv('data/metro.csv')
# Perfil dos Dados
data_profile(metro_data)

# Data profiling mostra que o arquivo tem 4 linhas de metrô,
# mas no Rio só existem 3 linhas: 1, 2, 4.
# Com isso, vamos mostrar na tela os valores únicos das linhas,
# para tentar descobrir o que está acontecendo.
secao("Valores únicos subway_line")
p(metro_data.subway_line.unique())

# O problema acontece porque existem espaços duplos no nome da linha 1.
# "Linha  1" (2 espaços) é a mesma entidade real que "Linha 1".
# Remover espaços duplos na coluna 'subway_line'
metro_data['subway_line'] = metro_data['subway_line'].str.replace('  ', ' ', regex=False)

# Verificar a correção conferindo os valores únicos na coluna 'subway_line'
secao("Valores únicos subway_line corrigida")
p(metro_data.subway_line.unique())

# Nome da linha resolvido.
# Data profiling mostra que vários valores de 'passengers' estão faltando.
# Vamos verificar se está relacionado à data.
missing_pax_by_month = metro_data.groupby('year_month').agg(
    missing_count=('passengers', lambda x: x.isnull().sum()),
    total_stations=('Station', 'nunique')
).reset_index()

# Exibir os resultados onde há valores ausentes em 'passengers'
secao("Análise passageiros x tempo") 
p(missing_pax_by_month[missing_pax_by_month['missing_count'] > 0]) 

# Todos os dados a partir de 2023-04-01 estão sem passageiros.
# Já podemos remover todas as linhas referentes ao ano de 2023 do conjunto de dados 'metro_data'
metro_data = metro_data[metro_data['year'] != 2023]

# Mesmo assim, existem várias outras linhas não relacionadas a isso.
# Tentando encontrar um padrão olhando para o detalhe de cada estação, o número de registros sem 'passengers' comparado com o total da estação.
# Analisar por estação
analyse_pax(metro_data)

# Essa análise mostra algumas estações com 12% e 8% de dados faltantes, o que é considerável.
# Vamos ver os detalhes de alguns desses casos.

list_station(metro_data,"Ipanema / General Osório")

list_station(metro_data,"São Conrado")

# Em ambos os casos, os dados iniciais de passageiros das estações começam com NaN.
# Hipótese: no ano de inauguração da estação foram incluídos registros desde o mês 1,
# porém sem valor de passageiros, pois a linha não estava operando. Eliminar essas linhas iniciais
metro_data = metro_data.sort_values(by=['Station', 'year_month']).reset_index(drop=True)
metro_data_cleaned = pd.DataFrame()  
for station in metro_data['Station'].unique():
    station_data = metro_data[metro_data['Station'] == station]
    first_non_null_index = station_data['passengers'].first_valid_index()
    if first_non_null_index is not None:
        station_data_cleaned = station_data.loc[first_non_null_index:]
        metro_data_cleaned = pd.concat([metro_data_cleaned, station_data_cleaned], ignore_index=True)
metro_data = metro_data_cleaned

# Analisar por estação novamente para ver se as mudanças tiveram efeito
analyse_pax(metro_data)

# Depois dessas intervenções, somente 9 registros, em apenas 1 estação, ficaram sem os dados de passageiros.
# Decidimos não preencher esses dados e aceitar que o movimento foi zero no período.
plot_boxplot(metro_data, 'year', 'passengers', 'Station', 4, "", "Pax / Mês", "BoxPlot - Distribuição de Passageiros Mensais. Categorizado por Estação, por Ano")

### Importacao e Analise de Dados de Populacao

# Ao ver o arquivo CSV, logo nas primeiras linhas, é fácil notar que o campo
# referente à população está mal formatado. Removendo espaços e vírgulas para
# importá-lo como número.
population_data = pd.read_csv('data/populacao.csv')
population_data['População'] = population_data['População'].replace({r'[^\d.]': ''}, regex=True).astype(float)

# Perfil dos Dados
data_profile(population_data)

# Data profile mostra 2 linhas sem informação. Quais são?
secao("Anos sem dados de população")
p(population_data[population_data['População'].isna()])

# Estimar os valores de população para os anos de 2022 e 2023 por interpolação linear
population_data = population_data.sort_values(by="Ano").reset_index(drop=True)
population_data['População'] = population_data['População'].interpolate(method='linear')

# Exibir as últimas linhas para confirmar a interpolação dos anos 2022 e 2023
secao("Interpolação anos 2022 e 2023")
p(population_data.tail(5))

# Como só temos dados de passageiros de 1998 a 2022, manter dados de população apenas nessa faixa
population_data = population_data[(population_data['Ano'] >= 1998) & (population_data['Ano'] <= 2022)]

### Importacao e Analise de Dados do PIB

# Carregar o arquivo CSV -- PIB usando ';' como delimitador
pib_data = pd.read_csv('data/pib.csv', delimiter=';')

# Perfil dos Dados
data_profile(pib_data)

# Data profile mais complexo nesse caso. Foi necessário abrir o arquivo para entender sua estrutura.
# O dado que queremos é composto pelas informações de 3 indicadores. Criar gráfico para entender melhor.
pib_indicadores = pib_data[pib_data['Nível'].isin(['1.1', '1.2', '1.3'])]

anos = pib_indicadores.columns[2:-1]
pib_indicador_1_1 = pib_indicadores[pib_indicadores['Nível'] == '1.1'].iloc[0, 2:-1].values.astype(float)
pib_indicador_1_2 = pib_indicadores[pib_indicadores['Nível'] == '1.2'].iloc[0, 2:-1].values.astype(float)
pib_indicador_1_3 = pib_indicadores[pib_indicadores['Nível'] == '1.3'].iloc[0, 2:-1].values.astype(float)

# Criar DataFrame
pib = pd.DataFrame({
    "Revisado": pib_indicador_1_1,
    "Retropolado": pib_indicador_1_2,
    "Encerrado": pib_indicador_1_3
}, index=anos.astype(int))

secao("PIB Inicial")
p(pib)
plot_line(pib, "Anos", "R$", "Linha - PIB por Ano - Séries Originais")

# Observamos uma regra para a série consolidada:
# 1.2 tem prioridade sobre 1.3
# 1.1 tem prioridade sobre 1.3
# Considerar 1.3 se for a única série com dados.
pib['Unificado'] = pib['Encerrado']
pib['Unificado'] = pib['Unificado'].where(pib['Retropolado'].isna(), pib['Retropolado'])
pib['Unificado'] = pib['Unificado'].where(pib['Revisado'].isna(), pib['Revisado'])

secao("PIB Unificado")
p(pib)
plot_line(pib[['Unificado']], "Anos", "R$", "Linha - PIB por Ano - Série Unificada")

### Consolidar dados

# Consolidar dados de Passageiros por ano
total_passengers_by_year = metro_data.groupby('year')['passengers'].sum()
total_passengers_by_year.index = total_passengers_by_year.index.astype(int)

# Consolidar dados de População por ano
population_by_year = population_data.set_index('Ano')['População']
population_by_year.index = population_by_year.index.astype(int)

# Combinar em um único DataFrame
combined_data = pd.DataFrame({
    'Passageiros': total_passengers_by_year,
    'PIB': pib['Unificado'],
    'População': population_by_year
})

secao("Dados Combinados - Todos os Anos")
p(combined_data)

# Análise gráfica dos dados combinados
plot_minmax(combined_data, "Ano", "Normalizado Min-Max", "Linha - Min-Max - Passageiros, PIB e População - Todos os Anos")

# Filtrar os dados para o período de 2001 a 2019 para análise de período "bem comportado"
start_year = 2001
end_year = 2019
filtered_data = combined_data[(combined_data.index >= start_year) & (combined_data.index <= end_year)]
string_years = f"{start_year} a {end_year}"

# Análise gráfica após correção
plot_minmax(filtered_data, "Ano", "Normalizado Min-Max", f"Linha - Min-Max - Passageiros, PIB e População - {string_years}")

# Gráficos de correlação
plot_correlation_heatmap(combined_data, "Matriz de Correlação - Passageiros, PIB e População - Todos os Anos")
plot_scatter(combined_data, 'Passageiros', 'População', "Scatter - Passageiros e População - Todos os Anos")
plot_scatter(combined_data, 'Passageiros', 'PIB', "Scatter - Passageiros e PIB - Todos os Anos")

plot_correlation_heatmap(filtered_data, f"Matriz de Correlação - Passageiros, PIB e População - {string_years}")
plot_scatter(filtered_data, 'Passageiros', 'População', f"Scatter - Passageiros e População - {string_years}")
plot_scatter(filtered_data, 'Passageiros', 'PIB', f"Scatter - Passageiros e PIB - {string_years}")

### Estimativa com rede neural

prediction(combined_data, split_year, epochs, batch_size)

print()
print("------------------------")
print("PROCESSAMENTO FINALIZADO")
print("------------------------")
