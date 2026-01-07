"""
GENERADOR DE CORPUS CORREGIDO PARA ANYTHINGLLM
Soluciona problemas de codificación y limpieza excesiva
"""

import pandas as pd
import re
import os

def limpiar_suave(texto):
    """Limpieza SUAVE sin destruir el texto"""
    if pd.isna(texto):
        return ""
    
    texto = str(texto).strip()
    
    # Solo normalizar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto)
    
    # Eliminar URLs
    texto = re.sub(r'http\S+|www.\S+', '', texto)
    
    # Eliminar menciones y hashtags duplicados
    texto = re.sub(r'(@\w+\s*)+', '@usuario ', texto)
    texto = re.sub(r'(#\w+\s*)+', '', texto)
    
    return texto.strip()


def crear_corpus_academico():
    """Corpus académico con codificación correcta"""
    
    academico = [
        {
            'fuente': 'Annie E. Casey Foundation',
            'tema': 'salud_mental',
            'tipo': 'estudio_academico',
            'texto': """84% de Gen Z cree que la salud mental es una crisis en Estados Unidos. 
            Son 80% más propensos a reportar ansiedad o depresión comparado con generaciones 
            anteriores. 42% de estudiantes de secundaria Gen Z reportaron sentimientos 
            persistentes de tristeza o desesperanza en 2021, casi 50% más alto que millennials 
            en los 2000s. Entre niñas, esta cifra fue 35% para millennials en 2001 comparado 
            con 57% de Gen Z en 2021."""
        },
        {
            'fuente': 'UNICEF Global Coalition',
            'tema': 'crisis_existencial',
            'tipo': 'estudio_academico',
            'texto': """6 de cada 10 Gen Z se sienten abrumados por eventos actuales. 
            Gen Z consume noticias más que cualquier otro contenido. 4 de cada 10 sienten 
            estigma al hablar de salud mental en escuelas y trabajos. 4 de cada 10 Gen Z 
            dicen necesitar apoyo para su salud mental."""
        },
        {
            'fuente': 'Emory University',
            'tema': 'redes_sociales',
            'tipo': 'estudio_academico',
            'texto': """Jóvenes que pasan más de 3 horas diarias en redes sociales tienen 
            mayor riesgo de problemas de salud mental. Gen Z ha estado expuesta a dispositivos 
            móviles, Wi-Fi de alta velocidad y redes sociales desde siempre. Un tercio de 
            adolescentes usa al menos un sitio casi constantemente. Algoritmos diseñados para 
            maximizar engagement son factor importante en daños de salud mental."""
        },
        {
            'fuente': 'ResearchGate',
            'tema': 'identidad_digital',
            'tipo': 'estudio_academico',
            'texto': """TikTok es usado predominantemente por Gen Z, con 63.5% de usuarios 
            menores de 29 años. Académicos describen TikTok como "crack digital". Los feeds 
            de video adictivos, curados por algoritmos, usan inteligencia artificial y minería 
            de datos para crear feed único para cada usuario. TikTok es capaz de moldear identidad."""
        },
        {
            'fuente': 'The Lemur Cultural Analysis',
            'tema': 'activismo_digital',
            'tipo': 'analisis_cultural',
            'texto': """Gen Z ha crecido con el peso del futuro inevitable. Redes sociales 
            amplifican activismo, por eso Gen Z ha tomado TikTok. El activismo impulsa innovación 
            lingüística. Desarrollo de "Algospeak" - uso de palabras código para subvertir 
            vigilancia del algoritmo. Gen Z enfrenta incertidumbre constante: COVID-19, ansiedad 
            climática, crisis de vivienda, inestabilidad global."""
        },
        {
            'fuente': 'PMC National Library Medicine',
            'tema': 'autenticidad',
            'tipo': 'estudio_academico',
            'texto': """Un usuario confesó: "El algoritmo de TikTok es perfecto, por eso paso 
            dos horas en él". Gen Z es cada vez más crítico de auto-presentación online, 
            frecuentemente percibida como falsa. BeReal se ha convertido en plataforma donde 
            usuarios prueban su "autenticidad" en contraste ideológico con Instagram o TikTok."""
        },
        {
            'fuente': 'Chaptly Teen Development',
            'tema': 'algoritmos_identidad',
            'tipo': 'analisis_psicologico',
            'texto': """Para Gen Z, significó crecer curado, siempre balanceando entre 
            autenticidad y performance. Para Gen Alpha, significa ser criados por algoritmos, 
            donde identidad es moldeada antes de ser completamente entendida. Gen Z fueron 
            primeros adolescentes en crecer completamente inmersos en mundo digital."""
        },
        {
            'fuente': 'Stanford University Press',
            'tema': 'burnout_rendimiento',
            'tipo': 'teoria_filosofica',
            'texto': """Byung-Chul Han: Nuestras sociedades competitivas están cobrando precio 
            al individuo tardo-moderno. En lugar de mejorar vida, multitasking y tecnología 
            están produciendo trastornos desde depresión hasta déficit de atención. Han interpreta 
            malestar como incapacidad para manejar experiencias negativas en era caracterizada 
            por positividad excesiva."""
        },
        {
            'fuente': 'Philosophy Break',
            'tema': 'autoexplotacion',
            'tipo': 'analisis_filosofico',
            'texto': """Han sugiere que sociedad moderna está atrapada por "imperativo de lograr" 
            y que burnout masivo es el resultado. Ya no somos "sujetos de obediencia" sino 
            "sujetos de rendimiento". Maximizamos nuestro propio rendimiento porque sentimos 
            que estamos ejerciendo nuestra libertad individual. El explotador es simultáneamente 
            el explotado. La auto-explotación es eficiente."""
        },
        {
            'fuente': 'Medium Philosophy Review',
            'tema': 'fatiga_social',
            'tipo': 'analisis_filosofico',
            'texto': """Byung-Chul Han nota que esta es historia de agotamiento, de fatiga 
            que individualiza y aísla. Autor describe enfermedades psíquicas como depresión 
            y burnout como consecuencias de incapacidad de decir no. Esta forma de vivir hace 
            al hombre equivalente a no-muerto, demasiado vivo y muerto para vivir. No tiene 
            tiempo para reflexionar, olvida su subjetividad."""
        }
    ]
    
    return pd.DataFrame(academico)


def generar_corpus_anythingllm():
    """Genera corpus corregido para AnythingLLM"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("="*80)
    print("GENERANDO CORPUS CORREGIDO PARA ANYTHINGLLM")
    print("="*80 + "\n")
    
    # 1. Cargar dataset sintético
    print("1. Cargando dataset sintético...")
    try:
        df_sintetico = pd.read_csv(os.path.join(script_dir, 'dataset_sintetico_5000_ampliado.csv'), encoding='utf-8')
        print(f"   {len(df_sintetico)} tweets cargados\n")
    except Exception as e:
        print(f"   Error: {e}\n")
        return
    
    # 2. Limpieza SUAVE (sin destruir palabras)
    print("2. Aplicando limpieza suave...")
    df_sintetico['texto_limpio'] = df_sintetico['texto'].apply(limpiar_suave)
    print("   Limpieza completada\n")
    
    # 3. Agregar corpus académico
    print("3. Agregando fuentes académicas...")
    df_academico = crear_corpus_academico()
    df_academico['texto_limpio'] = df_academico['texto'].apply(limpiar_suave)
    print(f"   {len(df_academico)} fuentes académicas agregadas\n")
    
    # 4. Combinar
    print("4. Combinando corpus...")
    df_sintetico['tipo'] = 'tweet_genz'
    df_sintetico['fuente'] = 'Twitter Gen Z'
    
    df_completo = pd.concat([
        df_sintetico[['texto_limpio', 'tema', 'fuente', 'tipo']],
        df_academico[['texto_limpio', 'tema', 'fuente', 'tipo']]
    ], ignore_index=True)
    
    print(f"   Total documentos: {len(df_completo)}\n")
    
    # 5. Guardar CSV limpio
    print("5. Guardando CSV...")
    df_completo.to_csv(os.path.join(script_dir, 'corpus_final_corregido.csv'), index=False, encoding='utf-8')
    print("   Guardado: corpus_final_corregido.csv\n")
    
    # 6. Crear archivo para AnythingLLM
    print("6. Creando archivo para AnythingLLM...")
    
    with open(os.path.join(script_dir, 'corpus_anythingllm_FINAL.txt'), 'w', encoding='utf-8') as f:
        # Escribir header
        f.write("CORPUS GEN Z: CRISIS DE SENTIDO EN LA ERA DIGITAL\n")
        f.write("="*80 + "\n")
        f.write("Análisis filosófico mediante RAG\n")
        f.write("Fuentes: Tweets Gen Z + Estudios Académicos\n")
        f.write("="*80 + "\n\n")
        
        for idx, row in df_completo.iterrows():
            # Formato narrativo estructurado
            doc = f"""
{'='*80}
DOCUMENTO {idx+1}
{'='*80}

FUENTE: {row['fuente']}
TEMA: {row['tema']}
TIPO: {row['tipo']}

CONTENIDO:
{row['texto_limpio']}

{'='*80}

"""
            f.write(doc)
    
    print("   Guardado: corpus_anythingllm_FINAL.txt\n")
    
    # 7. Estadísticas finales
    print("7. Estadísticas del corpus:")
    print(f"   Total documentos: {len(df_completo)}")
    print(f"   Tweets Gen Z: {len(df_sintetico)}")
    print(f"   Fuentes académicas: {len(df_academico)}")
    print(f"   Temas únicos: {df_completo['tema'].nunique()}")
    
    print("\n" + "="*80)
    print("CORPUS CORREGIDO GENERADO EXITOSAMENTE")
    print("="*80)
    print("\nARCHIVO PARA SUBIR A ANYTHINGLLM:")
    print("   corpus_anythingllm_FINAL.txt")
    print("\nEste archivo tiene:")
    print("   Codificación UTF-8 correcta")
    print("   Acentos y ñ preservados")
    print("   Texto legible y comprensible")
    print("   Estructura clara para RAG")
    
    return df_completo


# Ejecutar
if __name__ == "__main__":
    df = generar_corpus_anythingllm()