"""
main.py

Ponto de entrada do microsserviço FastAPI VigiaEnchente GCN.

Endpoints disponíveis nesta fase:
  GET /health  — liveness check para o orquestrador Spring Boot (porta 8080)

Endpoints planejados (Fase B):
  POST /risk/compute   — recebe vazão prevista, retorna scores de risco por nó
  POST /route/escape   — recebe origem/destino, retorna rota segura

Subir em desenvolvimento:
    uvicorn app.main:app --reload --port 8002
"""

from fastapi import FastAPI

app = FastAPI(
    title="VigiaEnchente — Serviço GCN",
    description="Microsserviço de predição de risco de enchente por nó viário usando GCN.",
    version="0.1.0",
)


@app.get("/health", tags=["infra"])
def health_check():
    """Liveness check — retorna 200 se o serviço está no ar."""
    return {"status": "ok"}
