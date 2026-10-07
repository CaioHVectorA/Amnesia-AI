# Amnesia AI

Amnesia AI é uma iniciativa de pesquisa e desenvolvimento que tem como objetivo criar um agente autônomo de inteligência artificial capaz de jogar Pokémon Competitivo em alto nível (visando Top 500 no ranking de Pokémon Showdown).

## 📚 Documentação do Projeto

- **[CONTEXT.md](CONTEXT.md)**: Arquitetura do sistema, modelagem do domínio do Pokémon competitivo, fluxo de dados e mapa de subprojetos.
- **[AGENTS.md](AGENTS.md)**: Diretrizes de engenharia, padrões de código e instruções para agentes de IA que colaboram no repositório.
- **[TODO.md](TODO.md)**: Roadmap detalhado e backlog de tarefas priorizado por fases.
- **[project.md](project.md)**: Motivação teórica, objetivos e estudo inicial de complexidade.

## 🗂️ Estrutura de Subprojetos

- `projects/simulator`: Motor de simulação rápida headless usando `@pkmn/sim` em TypeScript/Bun.
- `projects/type-classifier`: Modelo de classificação e aprendizado de características de Pokémon em Python (scikit-learn).
- `projects/matchup-calculator`: Calculador de vantagem de confrontos e matrizes de dano.
- `projects/moveset-recomendation`: Recomendador e preditor de movesets com base em estatísticas do Smogon.
- `projects/team-optimizer`: Otimizador e montador de times com análise de sinergia e cobertura de fraquezas.
- `data/`: Tabelas canônicas de dados do Showdown (`dex.json`, `moves.json`).