# USER MODULES
from config import PLOT, SAVE, SHOW
from user_text_lib import p, secao

# OTHER MODULES -- INSTALL DEPENDENCIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

def save_png(filename):
    # Salvar o gráfico como PNG
    if SAVE:
        filepath = "images/"+filename+".png"
        plt.savefig(filepath, format='png', dpi=300)  # Salvar o gráfico em alta resolução (300 dpi)

def show():
    if SHOW:
        plt.ion() 
        plt.show()
    else:
        plt.ioff()
        
def plot_scatter(data, x_column, y_column, title):
    """
    Função para criar um scatter plot entre duas colunas de um DataFrame e adicionar uma linha de regressão y = ax + b.
    
    Parâmetros:
    - data: DataFrame contendo os dados.
    - x_column: Nome da coluna para o eixo X.
    - y_column: Nome da coluna para o eixo Y.
    - title: Título do gráfico.
    """
    if PLOT:
        p("Scatter: "+title)
        plt.figure(figsize=(12, 6))
        
        # Scatter plot
        plt.scatter(data[x_column], data[y_column], color='blue', alpha=0.6, label='Dados')

        # Calcular a linha de regressão y = ax + b
        x = data[x_column]
        y = data[y_column]
        a, b = np.polyfit(x, y, 1)  # Ajuste de uma linha (grau 1)

        # Adicionar a linha de regressão ao gráfico
        plt.plot(x, a * x + b, color='red', linestyle='--', label=f'Linha de Regressão: y = {a:.2f}x + {b:.2f}')

        # Configurações do gráfico
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        # Salvar o gráfico como PNG
        save_png(title)
        show()

def plot_correlation_heatmap(data, title):
    """
    Função para criar um correlation heatmap matrix entre todas as colunas de um DataFrame.
    
    Parâmetros:
    - data: DataFrame contendo os dados.
    - title: Título do gráfico.
    """
    if PLOT:
        p("Correlation Heatmap Matrix: "+title)
        correlation_matrix = data.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, vmin=0, vmax=1)
        plt.title(title)
        save_png(title)
        show()

def plot_minmax(data,x_label,y_label,title):
    """
    Função para criar um gráfico de linhas, minmax no eixo y,
    com todas as séries de um DataFrame.
  
    Parâmetros:
    - data: DataFrame contendo os dados.
    - x_label: Label para o eixo X.
    - y_label: Label para o eixo Y.
    - title: Título do gráfico.
    """
    if PLOT:
        p("Line Min/Max: "+title)
        # Normalizar com Min-Max para cada serie
        scaler = MinMaxScaler()
        normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
        # Plotar cada coluna com labels
        plt.figure(figsize=(12, 6))
        for column in normalized_data.columns:
            plt.plot(normalized_data.index, normalized_data[column], label=column, marker='o')
        # Configurações
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.xticks(ticks=normalized_data.index, rotation=90)
        save_png(title)
        show()

def plot_line(data,x_label,y_label,title):
    """
    Cria um gráfico de linha para cada coluna do DataFrame 'data'.
    
    Parâmetros:
    data (DataFrame): DataFrame onde o índice representa os valores do eixo x e cada coluna representa uma série de dados.
    x_label
    y_label
    title
    """
    if PLOT:
        p("Line: "+title)
        plt.figure(figsize=(12, 6))
        
        # Plotar cada coluna como uma série no gráfico
        for column in data.columns:
            plt.plot(data.index, data[column], label=column, marker='o')
        
        # Configurações do gráfico
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.xticks(ticks=data.index,rotation=90)
        save_png(title)
        show()

def plot_boxplot(data, x_column, y_column, category, cols, x_label, y_label, title):
    """
    Cria gráficos de boxplot para cada categoria em um FacetGrid.
    
    Parâmetros:
    data (DataFrame): O DataFrame contendo os dados.
    x_column (str): Nome da coluna para o eixo x.
    y_column (str): Nome da coluna para o eixo y.
    category (str): Nome da coluna para separar os gráficos no FacetGrid.
    cols (int): Número de colunas no FacetGrid.
    x_label (str): Rótulo para o eixo x.
    y_label (str): Rótulo para o eixo y.
    title (str): Título do gráfico.
    """
    if PLOT:
        p("BoxPlot: "+title)

        # Definir uma lista ordenada para o eixo X
        x_order = sorted(data[x_column].unique())

        # Configurar as propriedades dos outliers (fliers)
        flierprops = dict(marker='o', markersize=1, linestyle='none')  # Adjust markersize to make dots smaller

        # Configurar a figura com FacetGrid para criar um gráfico de boxplot para cada categoria
        g = sns.FacetGrid(data, col=category, col_wrap=cols, height=4, sharex=False, sharey=False, despine=False)
        g.map_dataframe(sns.boxplot, x=x_column, y=y_column, palette="Oranges", hue=x_column, order=x_order, flierprops=flierprops)

        # Ajustar os rótulos e layout
        g.set_axis_labels(x_label, y_label)
        g.set_titles("{col_name}")
        g.fig.suptitle(title,y=1.01)

        # Rotacionar e reduzir o tamanho da fonte dos rótulos dos eixos x e y em todos os gráficos
        for ax in g.axes.flat:
            # Ajuste para os rótulos do eixo x
            for label in ax.get_xticklabels():
                label.set_rotation(90)
                label.set_fontsize(6)  # Ajustar o tamanho da fonte do eixo x para 8

            # Ajuste para os rótulos do eixo y
            for label in ax.get_yticklabels():
                label.set_fontsize(6)  # Ajustar o tamanho da fonte do eixo y para 8

        # Salvar como uma imagem de alta resolução sem abrir no Matplotlib
        if SAVE:
            file_path = "images/"+title+".png"
            g.savefig(file_path, dpi=300)

        # Abrir a imagem salva em um editor de imagem
        if SHOW:
            img = Image.open(file_path)
            img.show()

