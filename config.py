### ARQUIVO DE CONFIGURAÇÃO

## Configurações de Gráficos
# Define se os gráficos devem ser gerados
PLOT = True

# Define se os gráficos gerados devem ser exibidos na tela
SHOW = False

# Define se os gráficos gerados devem ser salvos em arquivos
SAVE = True

## Configurações da Rede Neural
# Habilita ou desabilita o treinamento e previsão da rede neural
PREDICTION = True

# Ano de corte para divisão dos dados entre treino e teste; dados de teste a partir do ano seguinte
split_year = 2015

# Número de épocas (repetições) para o treinamento da rede neural
epochs = 500

# Tamanho do batch (lote) de dados a ser processado em cada passo do treinamento
batch_size = 4

## Configurações de Detalhamento e Depuração
# Nível de detalhamento nas saídas de depuração:
# 0 = Nenhum output, 1 = Saída mínima (seções), 2 = Saída detalhada, 3 = Saída completa (para debug profundo)
DEBUG_LEVEL = 3
