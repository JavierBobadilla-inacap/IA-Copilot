import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Carga de datos
def load_data(filepath):
    """
    Carga un archivo CSV con datos necesarios para el sistema de recomendación.
    
    Args:
        filepath (str): Ruta del archivo CSV.
    
    Returns:
        pd.DataFrame: DataFrame con los datos cargados.
    """
    return pd.read_csv(filepath)

# Preprocesamiento de datos
def preprocess_data(data, feature_column):
    """
    Preprocesa los datos para generar una matriz de características basada en texto.
    
    Args:
        data (pd.DataFrame): DataFrame con los datos.
        feature_column (str): Nombre de la columna que contiene las características de texto.
    
    Returns:
        sparse matrix: Matriz dispersa con las características procesadas.
    """
    vectorizer = CountVectorizer()
    feature_matrix = vectorizer.fit_transform(data[feature_column])
    return feature_matrix

# Generación de recomendaciones
def generate_recommendations(data, feature_matrix, item_index, top_n=5):
    """
    Genera recomendaciones basadas en similitud de coseno.
    
    Args:
        data (pd.DataFrame): DataFrame con los datos originales.
        feature_matrix (sparse matrix): Matriz de características procesadas.
        item_index (int): Índice del ítem para el cual se generarán recomendaciones.
        top_n (int): Número de recomendaciones a generar.
    
    Returns:
        list: Lista de índices de los ítems recomendados.
    """
    cosine_sim = cosine_similarity(feature_matrix)
    similarity_scores = list(enumerate(cosine_sim[item_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_indices = [i[0] for i in similarity_scores[1:top_n+1]]
    return data.iloc[recommended_indices]

# Ejecución principal
if __name__ == "__main__":
    # Ruta de ejemplo para un archivo CSV
    filepath = "data.csv"
    
    # Cargar datos
    data = load_data(filepath)
    
    # Preprocesar datos
    feature_column = "description"  # Columna de ejemplo
    feature_matrix = preprocess_data(data, feature_column)
    
    # Generar recomendaciones
    item_index = 0  # Índice del ítem de ejemplo
    recommendations = generate_recommendations(data, feature_matrix, item_index)
    
    print("Recomendaciones:")
    print(recommendations)