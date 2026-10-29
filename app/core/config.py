"""
config.py

Configurações do microsserviço lidas de variáveis de ambiente.

Uso:
    from app.core.config import settings
    print(settings.risk_buffer_m)

Variáveis de ambiente reconhecidas (case-insensitive):
    RISK_BUFFER_M       — buffer de risco ao rio em metros (padrão: 50)
    PENALTY_FACTOR      — multiplicador de peso em nós de risco (padrão: 2.0)
    PLACE               — localidade OSM para download do grafo
    ML_SERVICE_URL      — URL do serviço ML que fornece a vazão prevista
    MODEL_PATH          — caminho para o arquivo .pt do modelo treinado
    GRAPH_CACHE_PATH    — pasta onde o grafo OSM em cache é armazenado

O arquivo .env na raiz do projeto é carregado automaticamente em
desenvolvimento. Em produção, prefira injetar as variáveis diretamente
no contêiner.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Risco ────────────────────────────────────────────────────────────────
    risk_buffer_m: float = 50.0
    """Distância máxima ao rio para classificar um nó como de risco (metros)."""

    penalty_factor: float = 2.0
    """Multiplicador aplicado ao peso de arestas adjacentes a nós de risco."""

    # ── Grafo ────────────────────────────────────────────────────────────────
    place: str = "Sabará, Minas Gerais, Brazil"
    """Localidade usada no download do grafo viário via OSMnx."""

    # ── Paths ────────────────────────────────────────────────────────────────
    model_path: Path = Path("models/gcn_risk.pt")
    """Caminho para o state_dict do modelo GCN treinado."""

    graph_cache_path: Path = Path("data/cache")
    """Pasta onde o grafo OSM em cache é salvo para evitar downloads repetidos."""

    # ── Integração ───────────────────────────────────────────────────────────
    ml_service_url: str = "http://localhost:8001"
    """URL base do serviço de ML que fornece a vazão prevista (m³/s)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Instância única compartilhada por toda a aplicação (singleton por módulo)
settings = Settings()
