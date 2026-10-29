"""
risk_service.py

Camada de serviço para cálculo de risco de enchente por nó da malha viária.

Responsabilidade futura (Fase B):
  - Carregar o grafo viário de Sabará uma única vez na inicialização
  - Carregar os pesos da GCN treinada (models/gcn_risk.pt)
  - Receber a vazão prevista em m³/s do serviço ML (porta 8001)
  - Calcular o score de risco por nó usando a GCN
  - Devolver um dict {node_id: risk_score} para os endpoints

Por que uma classe e não funções soltas?
  RiskService mantém estado (grafo carregado, modelo em memória). Carregar
  ambos a cada requisição seria inviável — o download do OSM leva ~10s e
  o forward da GCN precisa do grafo para montar edge_index. Manter estado
  em atributos de instância é a forma mais simples e direta de resolver isso.
"""


class RiskService:
    """Serviço de inferência de risco de enchente por nó.

    Instanciado uma única vez no startup do FastAPI (via lifespan ou
    injeção de dependência). Não implementado ainda — placeholder para Fase B.
    """

    def __init__(self):
        # TODO (Fase B): carregar grafo e modelo no __init__
        pass

    def compute_risk(self, flow_m3s: float) -> dict:
        """Calcula o score de risco para cada nó dado uma vazão prevista.

        Args:
            flow_m3s: vazão prevista em m³/s (recebida do serviço ML).

        Returns:
            dict mapeando node_id (int) → risk_score (float em [0, 1]).

        TODO (Fase B): implementar com GCN carregada no __init__.
        """
        raise NotImplementedError("RiskService.compute_risk será implementado na Fase B.")
