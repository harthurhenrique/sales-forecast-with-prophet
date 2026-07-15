from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import numpy as np
import os
import time
from prophet import Prophet
from sklearn.model_selection import ParameterGrid
from prophet.diagnostics import cross_validation, performance_metrics

def calcular_metricas_completas(df, col_real='y', col_pred='yhat', retorno_dict=False):
    # Remove as linhas onde 'y' ou 'yhat' sejam NaN
    df_limpo = df.dropna(subset=[col_real, col_pred])
    
    if df_limpo.empty:
        raise ValueError("Após remover os nulos, não sobrou nenhuma linha para avaliar!")
        
    y_real = df_limpo[col_real]
    y_pred = df_limpo[col_pred]
    
    # 1. MSE
    mse = mean_squared_error(y_real, y_pred)
    
    # 2. RMSE
    rmse = np.sqrt(mse)
    
    # 3. MAE
    mae = mean_absolute_error(y_real, y_pred)
    
    # 4. MAPE (Ajustado para não explodir com zeros)
    erro_perc_abs = np.where(
        y_real != 0, 
        np.abs((y_real - y_pred) / y_real), 
        np.nan 
    )
    mape = np.nanmean(erro_perc_abs)
    
    # 5. MDAPE 
    mdape = np.nanmedian(erro_perc_abs)
    
    # 6. SMAPE
    numerador = np.abs(y_pred - y_real)
    denominador = np.abs(y_real) + np.abs(y_pred)
    smape_linhas = np.divide(numerador, denominador, out=np.zeros_like(numerador), where=denominador!=0)
    smape = np.mean(smape_linhas) * 2 
    
    # ==========================================
    # RETORNO PARA O PIPELINE (DICIONÁRIO PURO)
    # ==========================================
    if retorno_dict:
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,   # Deixamos como float puro, sem multiplicar por 100
            'MDAPE': mdape,
            'SMAPE': smape
        }

    # ==========================================
    # FORMATAÇÃO PARA LEITURA HUMANA (SEU CÓDIGO ORIGINAL)
    # ==========================================
    resultados = {
        'Métrica': ['MSE', 'RMSE', 'MAE', 'MAPE', 'MDAPE', 'SMAPE'],
        'Valor (Formatado)': [
            f"{mse:,.2f}", 
            f"{rmse:,.2f}", 
            f"{mae:,.2f}", 
            f"{mape * 100:,.2f}%", 
            f"{mdape * 100:,.2f}%", 
            f"{smape * 100:,.2f}%"
        ]
    }
    
    return pd.DataFrame(resultados)

def extrair_feriados_prophet(df):
    """
    Percorre o DataFrame original e extrai eventos relevantes no formato do Prophet.
    Retorna um DataFrame com as colunas 'ds' e 'holiday'.
    """
    lista_eventos = []
    
    # 1. Regra das Lojas Fechadas (Open == 0)
    # if 'Open' in df.columns:
    #     df_fechada = df[df['Open'] == 0][['Date']].copy()
    #     df_fechada.rename(columns={'Date': 'ds'}, inplace=True)
    #     df_fechada['holiday'] = 'loja_fechada'
    #     lista_eventos.append(df_fechada)
        
    # 2. Regra dos Feriados Estaduais (StateHoliday ativo)
    if 'StateHoliday' in df.columns:
        mascara_feriado = (df['StateHoliday'] == 1) | (df['StateHoliday'] == '1') | df['StateHoliday'].isin(['a', 'b', 'c'])
        df_estado = df[mascara_feriado][['Date']].copy()
        df_estado.rename(columns={'Date': 'ds'}, inplace=True)
        df_estado['holiday'] = 'feriado_estadual'
        lista_eventos.append(df_estado)
        
    # 3. Regra dos Feriados Escolares (SchoolHoliday == 1)
    if 'SchoolHoliday' in df.columns:
        df_escola = df[df['SchoolHoliday'] == 1][['Date']].copy()
        df_escola.rename(columns={'Date': 'ds'}, inplace=True)
        df_escola['holiday'] = 'feriado_escolar'
        lista_eventos.append(df_escola)
        
    # 4. Regra de Promoções (Promo == 1) - Opcional, mas altamente recomendado
    # if 'Promo' in df.columns:
    #     df_promo = df[df['Promo'] == 1][['Date']].copy()
    #     df_promo.rename(columns={'Date': 'ds'}, inplace=True)
    #     df_promo['holiday'] = 'promocao'
    #     lista_eventos.append(df_promo)
        
    # Combina os DataFrames criados
    df_feriados_prophet = pd.concat(lista_eventos, ignore_index=True)
    
    # Converte para datetime e remove duplicatas exatas
    df_feriados_prophet['ds'] = pd.to_datetime(df_feriados_prophet['ds'])
    df_feriados_prophet.drop_duplicates(inplace=True)
    
    return df_feriados_prophet

def promover_modelo_campeao(df_bruto, parametros):
    """
    Função para realizar a promoção do modelo campeão, no caso o que tem menor MAPE.
    """
    periodo_teste = 30
    frequencia = 'D'

    # 1. Extração automatizada de feriados
    print("Mapeando feriados e eventos diretamente dos dados históricos...")
    df_feriados = extrair_feriados_prophet(df_bruto)

    # 2. Conversão e preparação (Padrão Prophet)
    print("Preparando features temporais...")
    df_prophet = df_bruto[['Date', 'Sales']].copy()
    df_prophet.rename(columns={'Date': 'ds', 'Sales': 'y'}, inplace=True)
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)

    # ==========================================
    # 3. Divisão Temporal Dinâmica (Train / Test Split)
    # ==========================================
    # Usamos o pd.to_timedelta para subtrair exatamente o tempo especificado da data máxima
    delta_tempo = pd.to_timedelta(periodo_teste, unit=frequencia)
    ponto_corte = df_prophet['ds'].max() - delta_tempo
    
    df_treino = df_prophet[df_prophet['ds'] <= ponto_corte]
    df_teste = df_prophet[df_prophet['ds'] > ponto_corte]
    
    print(f"Corte temporal estabelecido em: {ponto_corte.date()}")
    print(f"Treino: {df_treino.shape[0]} linhas | Teste: {df_teste.shape[0]} linhas")

    # 4. Instanciar e treinar o modelo campeão
    modelo = Prophet(**parametros, holidays=df_feriados)
    inicio_treino = time.time()
    modelo.fit(df_treino)
    fim_treino = time.time()
    tempo_execucao = round(fim_treino - inicio_treino, 2)

    print("Modelo treinado, gerando previsões...")
    # 5. Previsão dos próximos 30 dias.
    previsao = modelo.predict(df_teste[['ds']])
    df_avaliacao = df_teste[['ds', 'y']].merge(previsao[['ds', 'yhat', 'yhat_upper', 'yhat_lower']], on='ds', how='inner')

    # 6. Verifica as métricas.
    metricas = calcular_metricas_completas(df_avaliacao, col_real='y', col_pred='yhat', retorno_dict=True)
    print("Processo finalizado :)\nMétricas no modelo campeão:\n")
    print(metricas)

    return df_avaliacao

def executar_pipeline_otimizacao(df_bruto, grid_parametros, periodo_teste=30, frequencia='D', arquivo_log="../data/experimentos_prophet.csv"):
    """
    Pipeline automatizado para rodar múltiplos experimentos com o Prophet.
    O tamanho do conjunto de teste é definido dinamicamente pelos parâmetros 
    'periodo_teste' e 'frequencia'.
    """
    # 1. Extração automatizada de feriados
    print("Mapeando feriados e eventos diretamente dos dados históricos...")
    df_feriados = extrair_feriados_prophet(df_bruto)
    
    # 2. Conversão e preparação (Padrão Prophet)
    print("Preparando features temporais...")
    df_prophet = df_bruto[['Date', 'Sales']].copy()
    df_prophet.rename(columns={'Date': 'ds', 'Sales': 'y'}, inplace=True)
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)
    
    # ==========================================
    # 3. Divisão Temporal Dinâmica (Train / Test Split)
    # ==========================================
    # Usamos o pd.to_timedelta para subtrair exatamente o tempo especificado da data máxima
    delta_tempo = pd.to_timedelta(periodo_teste, unit=frequencia)
    ponto_corte = df_prophet['ds'].max() - delta_tempo
    
    df_treino = df_prophet[df_prophet['ds'] <= ponto_corte]
    df_teste = df_prophet[df_prophet['ds'] > ponto_corte]
    
    print(f"Corte temporal estabelecido em: {ponto_corte.date()}")
    print(f"Treino: {df_treino.shape[0]} linhas | Teste: {df_teste.shape[0]} linhas")
    
    # 4. Configuração do Grid de Hiperparâmetros
    grid = ParameterGrid(grid_parametros)
    total_experimentos = len(grid)
    print(f"Iniciando o Grid Search. Total de combinações a testar: {total_experimentos}\n")
    
    # 5. Loop de Execução dos Experimentos
    for i, params in enumerate(grid):
        print(f"[{i+1}/{total_experimentos}] Executando configuração: {params}")
        
        # Instanciar e treinar
        modelo = Prophet(**params, holidays=df_feriados)
        
        inicio_treino = time.time()
        modelo.fit(df_treino)
        fim_treino = time.time()
        tempo_execucao = round(fim_treino - inicio_treino, 2)
        
        # ==========================================
        # 6. Avaliação e Métricas
        # ==========================================
        # O modelo faz a previsão passando apenas o DataFrame com as datas do teste
        previsao = modelo.predict(df_teste[['ds']])
        
        # Cruzamos o valor real (y) com a previsão do Prophet (yhat) pela data (ds)
        df_avaliacao = df_teste[['ds', 'y']].merge(previsao[['ds', 'yhat']], on='ds', how='inner')
        
        # Chamamos a função passando as colunas e exigindo o dicionário
        metricas = calcular_metricas_completas(df_avaliacao, col_real='y', col_pred='yhat', retorno_dict=True)
        
        # Consolidação do registro da rodada
        registro = {
            'data_execucao': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metodo': 'Holdout',
            **params,
            **metricas,
            'tempo_execucao': tempo_execucao
        }
        
        # Salva incrementalmente no CSV
        df_registro = pd.DataFrame([registro])
        gravar_cabecalho = not os.path.exists(arquivo_log)
        df_registro.to_csv(arquivo_log, mode='a', header=gravar_cabecalho, index=False)
        
    print(f"\nPipeline concluído com sucesso! Histórico salvo em: {arquivo_log}")


def executar_pipeline_validacao_cruzada(df_bruto, grid_parametros, initial, period, horizon, arquivo_log="../data/experimentos_prophet_cv.csv"):
    """
    Pipeline automatizado para rodar múltiplos experimentos usando a Validação Cruzada
    e as métricas de performance nativas do Prophet.
    """
    # 1. Extração automatizada de feriados
    print("Mapeando feriados e eventos diretamente dos dados históricos...")
    df_feriados = extrair_feriados_prophet(df_bruto) 
    
    # 2. Conversão e preparação (Padrão Prophet)
    print("Preparando features temporais...")
    df_prophet = df_bruto[['Date', 'Sales', 'Open', 'Promo']].copy()
    df_prophet.rename(columns={'Date': 'ds', 'Sales': 'y', 'Open': 'open', 'Promo': 'promo'}, inplace=True)
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)
    
    print(f"Base total pronta para Validação Cruzada: {df_prophet.shape[0]} linhas preparadas.")
    
    # 3. Configuração do Grid de Hiperparâmetros
    grid = ParameterGrid(grid_parametros)
    total_experimentos = len(grid)
    print(f"Iniciando o Grid Search com Validação Cruzada. Total de combinações a testar: {total_experimentos}\n")
    
    # 4. Loop de Execução dos Experimentos
    for i, params in enumerate(grid):
        print(f"[{i+1}/{total_experimentos}] Executando configuração: {params}")
        
        # Instanciar o modelo
        modelo = Prophet(**params, holidays=df_feriados)
        modelo.add_regressor('open')
        modelo.add_regressor('promo')
        
        inicio_treino = time.time()
        
        # Treina o modelo na base de dados COMPLETA
        modelo.fit(df_prophet) 
        
        # ==========================================
        # 5. Avaliação e Métricas Nativas
        # ==========================================
        # Executa a janela deslizante
        df_cv = cross_validation(
            modelo, 
            initial=initial, 
            period=period, 
            horizon=horizon, 
            parallel="processes" # Acelera o processamento usando múltiplos núcleos
        )
        
        # Calcula as métricas extraindo a média do horizonte todo (rolling_window=1)
        # O Prophet retorna um DataFrame com: 'mse', 'rmse', 'mae', 'mape', 'mdape', 'smape', 'coverage'
        df_metricas = performance_metrics(df_cv, rolling_window=1)
        
        fim_treino = time.time()
        tempo_execucao = round(fim_treino - inicio_treino, 2)
        
        # Extraímos os valores da primeira (e única) linha do dataframe de métricas para um dicionário
        metricas = {
            'rmse': df_metricas['rmse'].values[0] if 'rmse' in df_metricas.columns else None,
            'mae': df_metricas['mae'].values[0] if 'mae' in df_metricas.columns else None,
            'mape': df_metricas['mape'].values[0] if 'mape' in df_metricas.columns else None,
            'mdape': df_metricas['mdape'].values[0] if 'mdape' in df_metricas.columns else None,
            'coverage': df_metricas['coverage'].values[0] if 'coverage' in df_metricas.columns else None
        }
        
        # 6. Consolidação do registro da rodada
        registro = {
            'data_execucao': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metodo': 'Validacao Cruzada',
            'parametros': str(params),
            **metricas,
            'tempo_execucao': tempo_execucao
        }
        
        # Salva incrementalmente no CSV
        df_registro = pd.DataFrame([registro])
        gravar_cabecalho = not os.path.exists(arquivo_log)
        df_registro.to_csv(arquivo_log, mode='a', header=gravar_cabecalho, index=False)
        
    print(f"\nPipeline de Validação Cruzada concluído com sucesso! Histórico salvo em: {arquivo_log}")
    return df_registro