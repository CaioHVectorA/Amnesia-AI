# Amnesia-AI v2

Amnesia-AI v2 é um agente autônomo de inteligência artificial de alta performance para Pokémon Showdown, baseado no artigo seminal da conferência IEEE CoG:

> **"A Self-Play Policy Optimization Approach to Battling Pokémon"**  
> *Dan Huang & Scott Lee — IEEE Conference on Games (CoG 2019)*  
> [Link para o Paper](https://www.yuzeh.com/assets/CoG-2019-Pkmn.pdf)

O objetivo central do projeto é alcançar **desempenho competitivo de nível mestre humano (>1600 Glicko-1 / Top Ladder)** no Pokémon Showdown exclusivamente através de **Reinforcement Learning (PPO), Generalized Advantage Estimation (GAE) e Self-Play Simétrico Puro**, sem árvores de busca (Minimax/MCTS) ou heurísticas engessadas.

---

## 📚 Documentação Central

- **[`CONTEXT.md`](CONTEXT.md)**: Fundamentação teórica completa, modelagem POMDP de batalhas simultâneas, Teoria dos Jogos e Equilíbrio de Nash, arquitetura de rede de 1.3M de parâmetros com embeddings de entidades (128-d) e post-mortem detalhado da v1.
- **[`AGENTS.md`](AGENTS.md)**: Manual operacional e padrões de engenharia para desenvolvimento com agentes de IA, incluindo as 4 regras invioláveis do projeto.
- **[`METAS.md`](METAS.md)**: Critérios quantitativos formais, intervalos de confiança de Wilson (95%), metas de win rate contra baselines e matriz de mitigação de anomalias.
- **[`TODO.md`](TODO.md)**: Roadmap atômico de execução fase por fase (Fases 1 a 5).

---

## 🏗️ Arquitetura do Pacote (`src/`)

```text
Amnesia-AI/
├── AGENTS.md                  # Manual de engenharia e regras de desenvolvimento
├── CONTEXT.md                 # Teoria do jogo, POMDP, arquitetura 1.3M e post-mortem
├── METAS.md                   # Metas estatísticas, Glicko-1 > 1600 e thresholds
├── TODO.md                    # Roadmap atômico fase por fase
├── requirements.txt           # Dependências mínimas essenciais
└── src/
    ├── __init__.py            # Raiz do pacote Amnesia-AI
    ├── model.py               # Rede Actor-Critic com Embeddings de 128-d e Action Masking
    ├── env.py                 # Wrapper do poke-env para extração de tensores
    ├── train.py               # Loop de Pure Self-Play PPO (Algoritmo 1 de Huang & Lee)
    ├── evaluate.py            # Avaliação de torneios contra Random, MaxDamage e Heuristic bots
    └── client.py              # Cliente de ladder online no Pokémon Showdown
```

---

## ⚡ Tecnologias

- **Python 3.12+**
- **PyTorch 2.6+** (CUDA 12.4 com aceleração TF32)
- **`poke-env` 0.16+** & **`gymnasium` 1.3+**
- **Formato Alvo**: `gen7randombattle` / `gen8randombattle` (níveis balanceados e diversidade procedural)