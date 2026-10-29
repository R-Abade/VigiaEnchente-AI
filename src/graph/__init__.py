"""
src/graph
─────────
Pacote central de modelagem de grafos viários para o VigiaEnchente.

Módulos
  loader        — carregamento e pré-processamento do grafo (OSMnx)
  flood         — risco de enchente: nós de risco, distância ao rio, penalização
  routing       — algoritmos de caminho mínimo
  features      — features de treino: betweenness, utm_to_latlon
  gcn           — modelo GCN, treino/avaliação e construção do dataset PyG
  visualization — funções de plot (saída em assets/images/)

Scripts orquestradores (não importar diretamente — use python -m scripts.*)
  scripts/vigiaEnchente    — pipeline Q4/Q5/Q6
  scripts/analysis_blocked — análise do grafo bloqueado
  scripts/train_gcn        — CLI de treino da GCN
  scripts/compare_models   — ablation study (Tabela 4 do artigo)
"""
