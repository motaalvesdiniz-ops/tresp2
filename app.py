# app.py — Monitor Cota de Gênero (Atualizado para langchain_classic)
# BigQuery + Gemini + FAISS + Clustering + Preditivo + Download PDF

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import io
import requests
from google.cloud import bigquery
from google.oauth2 import service_account
import os
from dotenv import load_dotenv
load_dotenv()


st.set_page_config(
    page_title="Monitor Cota de Gênero - TRE/SP",
    layout="wide",
    page_icon="⚖️"
)

os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'

# Texto / Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Gemini + Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Vectorstore FAISS
from langchain_community.vectorstores import FAISS

# IMPORTS ADAPTADOS: usar langchain_classic (compatibilidade com APIs antigas)
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# ML (clustering e predição)
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# =====================================================================
# Configurações iniciais
# =====================================================================
st.set_page_config(page_title="Monitor Cota de Gênero - TRE/SP",
                   layout="wide",
                   page_icon="⚖️")

st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }
    div.stButton > button { background-color: #4285F4; color: white; }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# Credenciais (substitua pelos seus caminhos/chaves)
# =====================================================================
KEY_PATH_BQ = r"C:\Users\Dell\Desktop\TRESP\tresp-479702-58079074efcc.json"
os.environ["GOOGLE_API_KEY"] = st.secrets["api_key"]
# =====================================================================
# Função: carregar dados do BigQuery
# =====================================================================
@st.cache_data(ttl=3600)
def load_data_from_bigquery():
    try:
        credentials = service_account.Credentials.from_service_account_file(KEY_PATH_BQ)
        client = bigquery.Client(credentials=credentials, project=credentials.project_id)

        query = """
        SELECT 
            id_processo, Relator, Data_Julgamento, Ementa, Texto_Completo, 
            link_pdf_gcs, ementa_resumida, tese_acusacao, tese_defesa, 
            local_eleicao, relator_ia, resultado_ia 
        FROM `tresp-479702.jurimetria_eleitoral.acordaos_cota_genero`
        """
        df = client.query(query).to_dataframe()
        df['Data_Julgamento'] = pd.to_datetime(df['Data_Julgamento'], errors='coerce')
        df['ano'] = df['Data_Julgamento'].dt.year.fillna(0).astype(int)
        # Padronizar colunas textuais
        for c in ['ementa_resumida', 'Texto_Completo', 'tese_defesa', 'ementa']:
            if c in df.columns:
                df[c] = df[c].fillna('').astype(str)
        df['Relator'] = df['Relator'].fillna('').astype(str)
        return df
    except Exception as e:
        st.error(f"Erro ao conectar no BigQuery: {e}")
        return pd.DataFrame()

# =====================================================================
# Setup: embeddings, vectorstore e chain (indexação)
# =====================================================================

from langchain_core.prompts import ChatPromptTemplate

@st.cache_resource
def setup_vectorstore_and_chain(df):
    if df.empty:
        return None, None, None

    # Criar campo de contexto
    df["contexto_ia"] = (
        "ID Processo: " + df['id_processo'].astype(str) + "\n" +
        "Relator: " + df['Relator'].astype(str) + "\n" +
        "Decisão (IA): " + df['resultado_ia'].astype(str) + "\n" +
        "Cidade: " + df['local_eleicao'].astype(str) + "\n" +
        "Resumo: " + df['ementa_resumida'].astype(str) + "\n" +
        "Tese Defesa: " + df['tese_defesa'].astype(str)
    )

    docs = df['contexto_ia'].fillna('').tolist()
    metadatas = df[['id_processo', 'link_pdf_gcs', 'Relator']].to_dict('records')

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = FAISS.from_texts(texts=docs, embedding=embeddings, metadatas=metadatas)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.2
    )

    # Criar o prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um assistente jurídico especializado em análise de casos eleitorais, "
                   "especificamente sobre fraude à cota de gênero. Use o contexto fornecido para "
                   "responder as perguntas de forma precisa e fundamentada."),
        ("system", "Contexto: {context}"),
        ("human", "{input}")
    ])

    # Criar chain de documentos
    doc_chain = create_stuff_documents_chain(llm, prompt)
    
    # Criar chain de recuperação (sem keyword argument)
    qa_chain = create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": 5}),
        doc_chain
    )

    return vectorstore, qa_chain, embeddings
# =====================================================================
# Util: tentar baixar PDF via link (retorna bytes ou None)
# =====================================================================
def try_download_pdf_bytes(url, timeout=15):
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None

# =====================================================================
# CLUSTERING: pega embeddings (via embed_documents) e roda KMeans
# =====================================================================
@st.cache_data(ttl=3600)
def compute_clusters(_embeddings_model, docs_texts, k=6, random_state=42):
    """
    Calcula clusters usando KMeans sobre embeddings dos documentos.
    
    Args:
        _embeddings_model: modelo de embeddings (não será hasheado pelo cache)
        docs_texts: lista de textos para criar embeddings
        k: número de clusters
        random_state: seed para reprodutibilidade
    
    Returns:
        tuple: (labels, embeddings_array)
    """
    try:
        # _embeddings_model deve ter método embed_documents(texts)
        embs = _embeddings_model.embed_documents(docs_texts)
        embs = np.array(embs)
        
        # Ajustar k se necessário
        if embs.shape[0] < k:
            k = max(1, embs.shape[0])
            
        # Executar KMeans
        km = KMeans(n_clusters=k, random_state=random_state)
        labels = km.fit_predict(embs)
        
        return labels, embs
    except Exception as e:
        st.warning(f"Erro ao calcular clusters: {e}")
        return np.zeros(len(docs_texts), dtype=int), None

# =====================================================================
# PREDITIVO: treina um classificador simples (TF-IDF + LR)
# =====================================================================

@st.cache_data(ttl=3600)
def train_predictive_model(df, text_field='ementa_resumida', target_field='resultado_ia'):
    """
    Treina um modelo preditivo simples usando TF-IDF + Logistic Regression.
    
    Args:
        df: DataFrame com os dados
        text_field: nome da coluna de texto
        target_field: nome da coluna alvo
    
    Returns:
        dict com model, vectorizer, accuracy e report, ou None se falhar
    """
    # Filtrar e preparar
    df_train = df[[text_field, target_field]].dropna()
    if df_train.empty:
        return None
    
    X = df_train[text_field].astype(str).tolist()
    y = df_train[target_field].astype(str).tolist()
    
    # Verificar se há dados suficientes
    if len(X) < 10:
        st.warning("Poucos dados para treinar modelo preditivo (mínimo 10 registros).")
        return None
    
    # Contar exemplos por classe
    from collections import Counter
    class_counts = Counter(y)
    
    # Filtrar classes com pelo menos 2 exemplos (necessário para stratify)
    min_samples = 2
    valid_classes = {cls for cls, count in class_counts.items() if count >= min_samples}
    
    if len(valid_classes) < 2:
        st.warning("Dados insuficientes: é necessário pelo menos 2 classes com 2+ exemplos cada.")
        return None
    
    # Filtrar dataset para manter apenas classes válidas
    mask = df_train[target_field].isin(valid_classes)
    df_train_filtered = df_train[mask]
    X = df_train_filtered[text_field].astype(str).tolist()
    y = df_train_filtered[target_field].astype(str).tolist()
    
    try:
        # TF-IDF
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
        X_vec = tfidf.fit_transform(X)
        
        # Verificar novamente após TF-IDF (pode remover algumas linhas)
        if X_vec.shape[0] < 10:
            st.warning("Dados insuficientes após vetorização TF-IDF.")
            return None
        
        # Decidir se usa stratify ou não
        class_counts_final = Counter(y)
        can_stratify = all(count >= 2 for count in class_counts_final.values())
        
        if can_stratify:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_vec, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            # Sem stratify se ainda houver classes com apenas 1 exemplo
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_vec, y, test_size=0.2, random_state=42
            )
            st.info("Modelo treinado sem stratification devido a classes com poucos exemplos.")
        
        # Treinar modelo
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_tr, y_tr)
        
        # Avaliar
        preds = model.predict(X_te)
        acc = accuracy_score(y_te, preds)
        report = classification_report(y_te, preds, output_dict=True, zero_division=0)
        
        return {
            "model": model, 
            "vectorizer": tfidf, 
            "accuracy": acc, 
            "report": report,
            "classes_used": list(valid_classes),
            "classes_removed": list(set(class_counts.keys()) - valid_classes)
        }
        
    except Exception as e:
        st.error(f"Erro ao treinar modelo preditivo: {e}")
        return None
# =====================================================================
# Carregar dados e indexar
# =====================================================================
with st.spinner("Conectando ao BigQuery e carregando base..."):
    df_raw = load_data_from_bigquery()

if df_raw.empty:
    st.warning("Nenhum dado disponível. Verifique a conexão ao BigQuery.")
    st.stop()

vectorstore, qa_chain, embeddings_model = setup_vectorstore_and_chain(df_raw)

# =====================================================================
# Barra lateral - filtros gerais
# =====================================================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google_2015_logo.svg/368px-Google_2015_logo.svg.png", width=100)
st.sidebar.title("Filtros")
anos = sorted(df_raw['ano'].unique(), reverse=True)
filtro_ano = st.sidebar.multiselect("Ano", anos, default=anos[:2])
filtro_resultado = st.sidebar.multiselect("Resultado", df_raw['resultado_ia'].unique(), default=None)
filtro_relator = st.sidebar.multiselect("Relator", sorted(df_raw['Relator'].unique())[:10], default=None)

df = df_raw.copy()
if filtro_ano:
    df = df[df['ano'].isin(filtro_ano)]
if filtro_resultado:
    df = df[df['resultado_ia'].isin(filtro_resultado)]
if filtro_relator:
    df = df[df['Relator'].isin(filtro_relator)]

# =====================================================================
# Calcular clusters (usar a base filtrada para visualização)
# =====================================================================
with st.spinner("Calculando clusters..."):
    docs_for_cluster = df['contexto_ia'].fillna('').tolist() if 'contexto_ia' in df.columns else df['ementa_resumida'].fillna('').tolist()
    cluster_k = st.sidebar.slider("Número de clusters (K)", 2, 12, 6)
    cluster_labels, doc_embeddings = compute_clusters(embeddings_model, docs_for_cluster, k=cluster_k)
    # anexar ao DF atual
    if len(cluster_labels) == len(df):
        df = df.reset_index(drop=True)
        df['cluster'] = cluster_labels
    else:
        df['cluster'] = 0

# =====================================================================
# Treinar modelo preditivo
# =====================================================================
with st.spinner("Treinando modelo preditivo..."):
    pred_info = train_predictive_model(df_raw, text_field='ementa_resumida', target_field='resultado_ia')

# =====================================================================
# Layout principal (tabs)
# =====================================================================
st.title("🗳️ Painel de Jurimetria: Fraude à Cota de Gênero — Avançado")
st.caption("BigQuery + Gemini + LangChain (compat) + FAISS + Clustering + Modelo Preditivo")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Estatísticas", "📄 Julgados (Download PDF)", "🧭 Clusters", "🤖 Assistente & Preditivo"])

# ---------------- TAB 1: Estatísticas e gráfico textual -----------------
with tab1:
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Visão Geral — Gráficos")
        # Decisões por tipo
        if 'resultado_ia' in df.columns:
            fig = px.pie(df, names='resultado_ia', hole=0.5, title="Distribuição de Resultados (IA)")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Tendência anual (contagem de julgados)")
        df_time = df.groupby('ano').size().reset_index(name='qtd').sort_values('ano')
        fig_line = px.line(df_time, x='ano', y='qtd', markers=True)
        st.plotly_chart(fig_line, use_container_width=True)

    with col2:
        st.subheader("Gráfico Textual — Resumo automático")
        # Geração textual: sumarizar estatísticas chave
        total = len(df)
        por_relator = df['Relator'].value_counts().head(5)
        top_cities = df['local_eleicao'].value_counts().head(5)
        texto = f"Total de julgados mostrados: {total}\n\nTop relatores:\n"
        for r, c in por_relator.items():
            texto += f"• {r}: {c} casos\n"
        texto += "\nTop cidades:\n"
        for cty, c in top_cities.items():
            texto += f"• {cty}: {c} casos\n"
        st.code(texto)

        # Mini sparkline (linha pequena)
        if not df_time.empty:
            spark = px.line(df_time, x='ano', y='qtd', title="Sparkline: Julgados por ano")
            spark.update_layout(height=150, margin=dict(t=30,b=10,l=10,r=10))
            st.plotly_chart(spark, use_container_width=True)

# ---------------- TAB 2: Julgados e Download PDF -----------------
with tab2:
    st.subheader("Base de Julgados (com botão de download do PDF)")
    st.info("Clique em 'Baixar PDF' para tentar baixar diretamente do link GCS. Se o link exigir autenticação, abra no navegador.")

    # Mostrar tabela com botão de download
    display_df = df[['Data_Julgamento','id_processo','local_eleicao','Relator','resultado_ia','ementa_resumida','cluster','link_pdf_gcs']].copy()
    display_df['Data_Julgamento'] = display_df['Data_Julgamento'].dt.date
    st.dataframe(display_df.rename(columns={'id_processo':'Processo','Relator':'Relator','resultado_ia':'Resultado','link_pdf_gcs':'Link PDF'}), use_container_width=True)

    st.markdown("---")
    st.write("Selecionar um processo para baixar o PDF")
    sel_proc = st.selectbox("Escolha o ID do processo", options=df['id_processo'].tolist())
    row = df[df['id_processo'] == sel_proc].iloc[0]
    pdf_link = row.get('link_pdf_gcs', '')

    if pdf_link:
        st.write(f"Link: {pdf_link}")
        # Tentar baixar
        with st.spinner("Tentando baixar PDF..."):
            pdf_bytes = try_download_pdf_bytes(pdf_link)
        if pdf_bytes:
            file_name = f"{sel_proc}.pdf"
            st.download_button("🔽 Baixar PDF", data=pdf_bytes, file_name=file_name, mime="application/pdf")
        else:
            st.warning("Não foi possível baixar automaticamente (link possivelmente privado). Clique abaixo para abrir no navegador.")
            st.markdown(f"[Abrir PDF no navegador]({pdf_link})", unsafe_allow_html=True)
    else:
        st.info("Nenhum link de PDF disponível para este processo.")

# ---------------- TAB 3: Clusters -----------------
with tab3:
    st.subheader("Clusters de Similaridade (via Embeddings)")
    st.write("Agrupei os julgados em clusters pelo conteúdo (embeddings). Use os filtros para explorar cada cluster.")
    st.markdown("### Visão geral dos clusters")
    cluster_counts = df['cluster'].value_counts().reset_index()
    cluster_counts.columns = ['cluster','qtd']
    figc = px.bar(cluster_counts, x='cluster', y='qtd', labels={'qtd':'Quantidade','cluster':'Cluster'})
    st.plotly_chart(figc, use_container_width=True)

    st.markdown("### Explorar cluster específico")
    sel_cluster = st.selectbox("Escolha o cluster", options=sorted(df['cluster'].unique()))
    df_cluster = df[df['cluster'] == sel_cluster]
    st.write(f"{len(df_cluster)} processos no cluster {sel_cluster}")
    st.dataframe(df_cluster[['id_processo','Data_Julgamento','Relator','resultado_ia','ementa_resumida']].reset_index(drop=True), use_container_width=True)

    # Mostrar top termos do cluster (TF-IDF quick)
    if not df_cluster.empty:
        tf = TfidfVectorizer(max_features=2000, stop_words='english')
        X = tf.fit_transform(df_cluster['ementa_resumida'].fillna(''))
        # média por termo
        term_means = np.asarray(X.mean(axis=0)).ravel()
        terms = np.array(tf.get_feature_names_out())
        top_idx = term_means.argsort()[::-1][:20]
        top_terms = terms[top_idx]
        st.markdown("**Top termos (TF-IDF) no cluster:**")
        st.write(", ".join(top_terms))

# ---------------- TAB 4: Assistente (QA) e Preditivo -----------------
with tab4:
    st.header("🤖 Assistente IA — Pergunte sobre os casos")
    st.info("A IA responde com base nos documentos carregados (top K documentos).")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    question = st.chat_input("Digite sua pergunta jurídica...")
    if question:
        st.session_state.messages.append({"role":"user","content":question})
        st.chat_message("user").write(question)

        if qa_chain:
            with st.chat_message("assistant"):
                with st.spinner("Consultando a base..."):
                    resp = qa_chain.invoke({"input": question})
                    # resp geralmente tem 'answer' e 'context' (depende da versão)
                    answer = resp.get("answer") or resp.get("result") or str(resp)
                    sources = resp.get("context") or resp.get("source_documents") or []
                    final = answer + "\n\n**Fontes consultadas:**"
                    for s in sources:
                        meta = getattr(s, "metadata", s)
                        mid = meta.get('id_processo') if isinstance(meta, dict) else None
                        rel = meta.get('Relator') if isinstance(meta, dict) else None
                        final += f"\n• Processo {mid} — Relator {rel}"
                    st.write(final)
                    st.session_state.messages.append({"role":"assistant","content":final})
        else:
            st.error("Chain de QA não está inicializada.")

    st.markdown("---")
    st.header("📈 Módulo Preditivo (modelo simples)")
    if pred_info is None:
        st.warning("Não foi possível treinar o modelo preditivo (dados insuficientes).")
    else:
        st.write(f"Precisão do modelo (test set): **{pred_info['accuracy']:.3f}**")
        # Mostrar relatório resumido
        rpt = pred_info['report']
        st.dataframe(pd.DataFrame(rpt).T)

        st.markdown("**Fazer predição manual** (cole a ementa resumida do caso):")
        text_input = st.text_area("Ementa resumida para prever Resultado", height=120)
        if st.button("Prever Resultado"):
            vec = pred_info['vectorizer'].transform([text_input])
            pred = pred_info['model'].predict(vec)[0]
            probs = pred_info['model'].predict_proba(vec)[0]
            # map classes and probs
            classes = pred_info['model'].classes_
            prob_str = "\n".join([f"{c}: {p:.3f}" for c, p in zip(classes, probs)])
            st.success(f"Predição: **{pred}**")
            st.code(prob_str)

# =====================================================================
# Footer / dicas
# =====================================================================
st.sidebar.markdown("---")
st.sidebar.write("Dicas:")
st.sidebar.write("- Se o download automático falhar, o PDF pode exigir credenciais. Abra pelo link e faça o download manualmente.")
st.sidebar.write("- Para melhorar clusters, aumente K ou faça limpeza prévia dos textos.")
st.sidebar.write("- O modelo preditivo é básico; para produção recomendo features adicionais e validação robusta.")

