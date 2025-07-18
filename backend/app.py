from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO, StringIO
import joblib
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import logging

app = FastAPI(title="OCEAN Personality Analyzer API",
              description="API para análisis de personalidad usando el modelo OCEAN y K-Means",
              version="1.0.0")

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelo Pydantic para configuración
class AnalysisConfig(BaseModel):
    clusters: int = 5
    features: List[str]
    scale: bool = True
    pca_components: Optional[int] = None

# Variables globales para almacenar estado
current_analysis = {
    'dataset': None,
    'results': None,
    'config': None,
    'model': None
}

# ENDPOINTS PRINCIPALES

@app.post("/upload/")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        # Leer el archivo subido
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(400, "Formato de archivo no soportado")
        
        # Validar estructura básica del dataset
        required_columns = ['EXT1', 'EXT2', 'EXT3', 'EXT4', 'EXT5', 'EXT6', 'EXT7', 'EXT8', 'EXT9', 'EXT10',
                          'EST1', 'EST2', 'EST3', 'EST4', 'EST5', 'EST6', 'EST7', 'EST8', 'EST9', 'EST10',
                          'AGR1', 'AGR2', 'AGR3', 'AGR4', 'AGR5', 'AGR6', 'AGR7', 'AGR8', 'AGR9', 'AGR10',
                          'CSN1', 'CSN2', 'CSN3', 'CSN4', 'CSN5', 'CSN6', 'CSN7', 'CSN8', 'CSN9', 'CSN10',
                          'OPN1', 'OPN2', 'OPN3', 'OPN4', 'OPN5', 'OPN6', 'OPN7', 'OPN8', 'OPN9', 'OPN10']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise HTTPException(400, f"Faltan columnas requeridas: {missing_cols}")
        
        # Almacenar el dataset
        current_analysis['dataset'] = df
        current_analysis['results'] = None
        current_analysis['model'] = None
        
        # Obtener estadísticas descriptivas
        stats = {
            'summary': df.describe().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'columns': list(df.columns)
        }
        
        return {
            "message": "Dataset cargado exitosamente",
            "stats": stats,
            "preview": df.head().to_dict(orient='records')
        }
        
    except Exception as e:
        logger.error(f"Error al cargar dataset: {str(e)}")
        raise HTTPException(500, f"Error al procesar el archivo: {str(e)}")

@app.post("/analyze/")
async def analyze_data(config: AnalysisConfig):
    try:
        if current_analysis['dataset'] is None:
            raise HTTPException(400, "Primero cargue un dataset")
        
        df = current_analysis['dataset'].copy()
        features = config.features
        
        # Validar columnas seleccionadas
        invalid_cols = [col for col in features if col not in df.columns]
        if invalid_cols:
            raise HTTPException(400, f"Columnas no válidas: {invalid_cols}")
        
        # Preprocesamiento
        X = df[features]
        
        if config.scale:
            scaler = MinMaxScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=features)
        
        # Aplicar PCA si se especifica
        if config.pca_components and config.pca_components > 0:
            pca = PCA(n_components=config.pca_components)
            X = pd.DataFrame(pca.fit_transform(X), 
                            columns=[f"PC{i+1}" for i in range(config.pca_components)])
        
        # Entrenar modelo K-Means
        kmeans = KMeans(n_clusters=config.clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Almacenar resultados
        results_df = df.copy()
        results_df['cluster'] = clusters
        results_df['cluster'] = results_df['cluster'].astype(str)  # Para visualización
        
        # Calcular estadísticas por cluster
        cluster_stats = {}
        for trait in ['EXT', 'EST', 'AGR', 'CSN', 'OPN']:
            trait_cols = [col for col in df.columns if col.startswith(trait)]
            trait_scores = results_df.groupby('cluster')[trait_cols].mean().mean(axis=1)
            cluster_stats[trait] = trait_scores.to_dict()
        
        # Guardar resultados
        current_analysis['results'] = {
            'data': results_df,
            'stats': cluster_stats,
            'model': kmeans,
            'config': config
        }
        
        return {
            "message": "Análisis completado exitosamente",
            "cluster_counts": results_df['cluster'].value_counts().to_dict(),
            "cluster_stats": cluster_stats
        }
        
    except Exception as e:
        logger.error(f"Error en análisis: {str(e)}")
        raise HTTPException(500, f"Error en análisis: {str(e)}")

@app.get("/results/")
async def get_results():
    if current_analysis['results'] is None:
        raise HTTPException(400, "No hay resultados disponibles")
    
    results = current_analysis['results']
    sample_data = results['data'].head(50).to_dict(orient='records')
    
    return {
        "config": results['config'],
        "sample_data": sample_data,
        "cluster_stats": results['stats'],
        "cluster_counts": results['data']['cluster'].value_counts().to_dict()
    }

@app.get("/export/excel")
async def export_excel():
    if current_analysis['results'] is None:
        raise HTTPException(400, "No hay resultados para exportar")
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        current_analysis['results']['data'].to_excel(writer, index=False, sheet_name='Resultados')
        
        # Hoja de estadísticas
        stats_data = []
        for trait, values in current_analysis['results']['stats'].items():
            for cluster, score in values.items():
                stats_data.append({
                    'Trait': trait,
                    'Cluster': cluster,
                    'Score': score
                })
        pd.DataFrame(stats_data).to_excel(writer, index=False, sheet_name='Estadisticas')
    
    output.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ocean_results_{timestamp}.xlsx"
    
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/export/pdf")
async def export_pdf():
    if current_analysis['results'] is None:
        raise HTTPException(400, "No hay resultados para exportar")
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Título
    title = Paragraph("Resultados de Análisis OCEAN", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))
    
    # Configuración
    config = current_analysis['results']['config']
    config_text = f"<b>Configuración:</b> Clusters: {config.clusters}, Variables: {', '.join(config.features)}, Escalado: {config.scale}, PCA: {config.pca_components if config.pca_components else 'No'}"
    elements.append(Paragraph(config_text, styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Conteo de clusters
    counts = current_analysis['results']['data']['cluster'].value_counts().to_dict()
    count_data = [["Cluster", "Cantidad"]] + [[k, v] for k, v in counts.items()]
    
    count_table = Table(count_data)
    count_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(count_table)
    elements.append(Spacer(1, 24))
    
    # Estadísticas por cluster
    stats = current_analysis['results']['stats']
    stats_data = [["Cluster", "Extroversión", "Neuroticismo", "Amabilidad", "Responsabilidad", "Apertura"]]
    
    clusters = sorted(counts.keys())
    for cluster in clusters:
        row = [cluster]
        for trait in ['EXT', 'EST', 'AGR', 'CSN', 'OPN']:
            row.append(round(stats[trait][cluster], 2))
        stats_data.append(row)
    
    stats_table = Table(stats_data)
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(Paragraph("<b>Estadísticas por Cluster</b>", styles['Heading2']))
    elements.append(Spacer(1, 12))
    elements.append(stats_table)
    
    doc.build(elements)
    buffer.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ocean_results_{timestamp}.pdf"
    
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/plot/cluster_distribution")
async def plot_cluster_distribution():
    if current_analysis['results'] is None:
        raise HTTPException(400, "No hay resultados disponibles")
    
    df = current_analysis['results']['data']
    plt.figure(figsize=(10, 6))
    sns.countplot(x='cluster', data=df, palette='viridis')
    plt.title('Distribución de Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Cantidad')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    return StreamingResponse(buffer, media_type="image/png")

@app.get("/plot/trait_distribution")
async def plot_trait_distribution(trait: str):
    if current_analysis['results'] is None:
        raise HTTPException(400, "No hay resultados disponibles")
    
    valid_traits = ['EXT', 'EST', 'AGR', 'CSN', 'OPN']
    if trait not in valid_traits:
        raise HTTPException(400, f"Trait debe ser uno de: {', '.join(valid_traits)}")
    
    df = current_analysis['results']['data']
    trait_cols = [col for col in df.columns if col.startswith(trait)]
    
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(trait_cols, 1):
        plt.subplot(2, 5, i)
        sns.histplot(df[col], bins=20, kde=True)
        plt.title(col)
        plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    return StreamingResponse(buffer, media_type="image/png")

@app.get("/plot/pca")
async def plot_pca():
    if current_analysis['results'] is None:
        raise HTTPException(400, "No hay resultados disponibles")
    
    df = current_analysis['results']['data']
    features = current_analysis['results']['config'].features
    
    # Aplicar PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(df[features])
    
    # Crear gráfico
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=components[:, 0], y=components[:, 1], 
                    hue=df['cluster'], palette='viridis', alpha=0.7)
    plt.title('Visualización PCA de Clusters')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend(title='Cluster')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    return StreamingResponse(buffer, media_type="image/png")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)