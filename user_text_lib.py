# Módulos do Usuário
from config import DEBUG_LEVEL

# Importação de outros módulos necessários
import pandas as pd

def secao(texto):
    """
    Exibe uma seção formatada com o texto informado, delimitada por linhas, se o DEBUG_LEVEL for 1 ou superior.
    
    Parâmetros:
        - texto (str): Texto da seção a ser exibido.
    """
    if DEBUG_LEVEL >= 1:
        print("\n" + "-" * len(texto))
        print(texto)
        print("-" * len(texto))

def p(texto):
    """
    Exibe o texto informado, apenas se o DEBUG_LEVEL for 2 ou superior.

    Parâmetros:
        - texto (str): Texto a ser exibido.
    """
    if DEBUG_LEVEL >= 2:
        print("")
        print(texto)

def list_station(metro_data, station):
    """
    Exibe todos os registros da estação especificada, incluindo as colunas 'year_month' e 'passengers'.

    Parâmetros:
        - metro_data (DataFrame): DataFrame com os dados do metrô.
        - station (str): Nome da estação a ser listada.
    """
    pd.set_option('display.max_rows', None)  # Configurar para exibir todas as linhas
    secao("Dados da estação " + station)
    
    # Filtrar dados da estação e exibir apenas as colunas relevantes
    station_data = metro_data[metro_data['Station'] == station][['year_month', 'passengers']]
    p(station_data)
