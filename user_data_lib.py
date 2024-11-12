# Importação dos módulos do usuário
from config import PREDICTION, DEBUG_LEVEL

# Importação de outros módulos necessários
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from user_text_lib import secao, p
from user_graph_lib import plot_line

def prediction(input_data, split_year, epocas, batch_size):
    """
    Função para prever dados futuros usando uma rede neural.
    
    Parâmetros:
        - input_data (DataFrame): Dados de entrada para a previsão.
        - split_year (int): Ano para divisão entre dados de treino e teste.
        - epocas (int): Número de épocas para o treinamento.
        - batch_size (int): Tamanho do batch para o treinamento.
    """
    if PREDICTION:
        secao(f"REDE NEURAL -- Ano de divisão: {split_year} | Épocas: {epocas} | Batch Size: {batch_size}")

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Input, Dropout
        from tensorflow.keras.optimizers import Adam

        # Limpar dados, removendo linhas com valores NaN
        input_data.dropna(inplace=True)

        # Dividir dados entre treino e teste com base no ano definido
        train_data = input_data[input_data.index <= split_year]
        test_data = input_data[input_data.index > split_year]

        # Exibir conjuntos de treino e teste
        p("\nTreinamento:")
        p(train_data)
        p("\nTeste:")
        p(test_data)

        # Normalizar os dados usando Min-Max
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)

        # Preparar entradas (X) e saídas (y) para o treinamento
        X_train, y_train = train_scaled[:-1], train_scaled[1:]

        p("\nEntradas de Treinamento (X_train):")
        p(X_train)
        p("Saídas de Treinamento (y_train):")
        p(y_train)

        # Configurar a estrutura da rede neural
        model = Sequential([
            Input(shape=(X_train.shape[1],)),  # Definir a camada de entrada
            Dense(64, activation='relu'),      # Primeira camada oculta
            Dropout(0.2),                      # Dropout para reduzir overfitting
            Dense(32, activation='relu'),      # Segunda camada oculta
            Dense(3)                           # Camada de saída com 3 neurônios para Passageiros, PIB e População
        ])

        # Compilar o modelo com o otimizador Adam e uma taxa de aprendizado baixa
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        # Treinar o modelo com os dados de treino
        p("\nIniciando o treinamento do modelo...")
        model.fit(X_train, y_train, epochs=epocas, batch_size=batch_size, verbose=0)

        # Fazer previsões e desnormalizar os resultados
        predictions_scaled = model.predict(test_scaled)
        predictions = scaler.inverse_transform(predictions_scaled)

        p("\nPrevisões Desnormalizadas (predictions):")
        p(predictions)

        # Obter os valores reais para comparação
        actual_values = test_data.values
        p("\nValores Reais (actual_values):")
        p(actual_values)

        # Criar um DataFrame consolidado para plotar dados reais e previstos
        def preparar_dados_plot_line(input_data, test_data, predictions):
            years_test = test_data.index
            predictions_df = pd.DataFrame(predictions, index=years_test, columns=['Passageiros_Previsto', 'PIB_Previsto', 'População_Previsto'])
            combined_consolidado = pd.concat([input_data, predictions_df], axis=1)
            return combined_consolidado

        # Consolidar dados
        combined_consolidado = preparar_dados_plot_line(input_data, test_data, predictions)

        # Gerar gráficos comparativos para cada variável
        variables = ['Passageiros', 'PIB', 'População']
        for var in variables:
            data_plot = combined_consolidado[[var, f"{var}_Previsto"]]
            title = f"Previsão Rede Neural - {var} - (Epocas {epocas} - Batch {batch_size})"
            plot_line(data_plot, x_label="Ano", y_label=var, title=title)

def data_profile(data):
    """
    Função para exibir um perfil detalhado dos dados para análise exploratória.
    
    Parâmetros:
        - data (DataFrame): Dados a serem analisados.
    """
    if DEBUG_LEVEL == 3:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        secao("Data profiling")

        # Exibir informações sobre a estrutura dos dados
        p("Estrutura dos dados")
        p(data.info())

        # Mostrar as primeiras linhas dos dados
        p("\nPrimeiras linhas dos dados")
        p(data.head())

        # Verificar a presença de valores nulos
        p("\nValores faltantes em cada coluna")
        p(data.isnull().sum())

        # Exibir a descrição estatística dos dados
        p("\nDescrição estatística dos dados")
        p(data.describe(include='all'))

def analyse_pax(metro_data):
    """
    Realiza uma análise de dados faltantes de passageiros por estação,
    calculando a porcentagem de valores ausentes.

    Parâmetros:
        - metro_data (DataFrame): DataFrame contendo os dados do metrô, incluindo colunas 'Station' e 'passengers'.
    
    Esta função exibe:
        - O total de valores ausentes ('missing_count') de passageiros por estação.
        - A contagem total de registros por estação ('total_count').
        - A porcentagem de valores ausentes por estação ('missing_percentage').
    """
    secao("Análise passageiros x estação")
    
    # Contar o número de valores ausentes em 'passengers' por estação
    missing_pax_by_station = metro_data.groupby('Station')['passengers'].apply(lambda x: x.isnull().sum()).reset_index(name='missing_count')
    
    # Adicionar a contagem total de registros por estação
    missing_pax_by_station['total_count'] = metro_data.groupby('Station')['passengers'].size().values
    
    # Calcular a porcentagem de valores ausentes por estação
    missing_pax_by_station['missing_percentage'] = (missing_pax_by_station['missing_count'] / missing_pax_by_station['total_count']) * 100
    
    # Exibir estações com valores ausentes, ordenando pela maior porcentagem de faltantes
    p(missing_pax_by_station[missing_pax_by_station['missing_count'] > 0].sort_values(by='missing_percentage', ascending=False))

