# Pipeline RAG para Extração de características de Laudos de Mamografia

Mini projeto de **RAG (Retrieval-Augmented Generation)** construído com **LangChain** e **LangGraph** para extrair, de forma estruturada, achados clínicos de laudos de mamografia (cistos, nódulos, calcificações, microcalcificações e classificação BI-RADS).

> **Aviso:** este projeto tem fins **educacionais e experimentais**.

---

## 📌 Visão geral

O pipeline indexa laudos em PDF em um banco vetorial e, a partir de um novo laudo informado como pergunta, recupera os trechos mais semelhantes da base e usa um LLM para extrair os achados em um formato padronizado, seguindo regras de interpretação definidas no prompt.

### Informações extraídas

| Achado | Informação retornada |
|---|---|
| Cisto | Presente/Ausente, localização e tamanho |
| Nódulo | Presente/Ausente, localização e tamanho |
| Calcificação | Presente/Ausente, localização e tamanho |
| Microcalcificação | Presente/Ausente, localização e tamanho |
| BI-RADS | Categoria |
| Outras citações | Observações adicionais relevantes |

Quando uma informação não aparece no texto, o modelo retorna `[sem referência no texto]`.

---

## 🏗️ Arquitetura

O fluxo é dividido em duas etapas: **indexação** (feita uma vez) e **recuperação + geração** (a cada consulta).

```
INDEXAÇÃO
PDFs de laudos ──► PyPDFDirectoryLoader ──► RecursiveCharacterTextSplitter ──► Embeddings ──► Chroma

RECUPERAÇÃO + GERAÇÃO (LangGraph)
Pergunta (laudo) ──► retrieve (similarity search) ──► generate (Prompt + LLM) ──► Resposta estruturada
```

### Componentes

| Etapa | Tecnologia |
|---|---|
| Orquestração | [LangGraph](https://langchain-ai.github.io/langgraph/) (`StateGraph`) |
| Carregamento de documentos | `PyPDFDirectoryLoader` (LangChain Community) |
| Divisão em chunks | `RecursiveCharacterTextSplitter` (1000 caracteres, overlap de 200) |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` (Hugging Face, open source) |
| Banco vetorial | [Chroma](https://www.trychroma.com/) com persistência em diretório local |
| LLM | `mistral-small-2503` (Mistral AI) |
| Observabilidade | [LangSmith](https://smith.langchain.com/) (tracing) |

### Estado do grafo

O LangGraph passa um objeto `State` (`NamedTuple`) entre os nós:

```python
class State(NamedTuple):
    question: str                  # Pergunta/laudo informado pelo usuário
    context: Tuple[Document, ...]  # Trechos recuperados do banco vetorial
    answer: str                    # Resposta gerada pelo LLM
```

### Nós do grafo

1. **`retrieve`**: faz uma busca por similaridade no Chroma e retorna os trechos mais próximos da pergunta.
2. **`generate`**: concatena os trechos recuperados, monta o prompt customizado e chama o LLM.

---

## 🧠 Prompt e regras de interpretação

O prompt posiciona o modelo como especialista em análise de laudos de mamografia e inclui diretrizes específicas para reduzir ambiguidades:

- **Nódulo vs. cisto:** se um achado é descrito como nódulo na mamografia, mas confirmado como cisto no ultrassom, é classificado apenas como **cisto**. Complexos sólido-císticos são reportados em **ambas** as categorias.
- **Múltiplos achados:** todos são reportados, priorizando os classificados como suspeitos, os de maior tamanho e os com características atípicas.
- **Calcificações vs. microcalcificações:** a classificação segue o termo usado no laudo (por exemplo, "grosseiras" ou "vasculares" para calcificações; "puntiformes", "pleomórficas" ou "em cluster" para microcalcificações).

---

## 📂 Estrutura do projeto

```
.
├── LangChain_RAG.ipynb    # Notebook com o pipeline completo
├── RAG_exames/            # PDFs dos laudos a serem indexados (não versionar)
├── rag_chroma_db/         # Banco vetorial gerado pelo Chroma (não versionar)
├── .env                   # Chaves de API (não versionar)
└── README.md
```

---

### Exemplo de uso

```python
state_inicial = State(
    question="Faça a extração das características do seguinte exame mamográfico. ...",
    context=tuple([]),
    answer=" "
)

response = graph.invoke(state_inicial)
print(response["answer"])
```

O notebook também permite inspecionar o contexto recuperado (`response["context"]`) e visualizar o grafo do LangGraph em diagrama Mermaid.

---
