# Replay Scraper & Clustering Engine

Módulo de alta vazão para coleta, filtragem e clusterização de replays do Pokémon Showdown para treinamento de Behavioral Cloning (BC).

---

## ⚡ Recursos

- **Multi-threaded & Rate-limited**: Coleta centenas de replays por minuto com backoff automático e sem estourar limites de requisição.
- **Indexação em SQLite**: Deduplicação instantânea (`data/replays/replays_metadata.sqlite`), registrando metadados de jogadores, ratings, vencedores, número de turnos e status de desistência (*forfeit*).
- **Clusterização por Elo**: Particionamento automático dos arquivos em pastas de acordo com o nível competitivo:
  - `elite_1800plus/` (Elo $\ge 1800$)
  - `high_1650_1799/` (Elo $1650 - 1799$)
  - `mid_1500_1649/` (Elo $1500 - 1649$)
  - `low_1300_1499/` (Elo $1300 - 1499$)
  - `sub1300/` (Elo $< 1300$)
- **Pesagem de Amostras para BC**:
  - Calcula o peso $w$ de cada partida baseado no Elo do vencedor e penalidade para desistências instantâneas ($< 4$ turnos).

---

## 🚀 Como Usar

### 1. Coletar Replays em Larga Escala
```bash
# Coletar 1000 replays de Random Battle filtrando partidas de alto Elo
python projects/replay-scraper/scraper.py --format gen9randombattle --min-rating 1600 --max-replays 1000 --pages 25 --concurrency 6

# Coletar replays de Gen 9 OU
python projects/replay-scraper/scraper.py --format gen9ou --min-rating 1700 --max-replays 500 --pages 15
```

### 2. Analisar o Dataset Coletado
```bash
python projects/replay-scraper/analyze_replays.py
```
