# Representação de Estados e Vetorização de Batalha (State Representation)

Este documento especifica formalmente a conversão de um estado de batalha do Pokémon Showdown em tensores numéricos para redes neurais, algoritmos de Behavioral Cloning (BC) e avaliação de função de valor.

---

## 1. Princípios de Codificação

Para garantir máxima eficiência no treinamento offline e inferência em tempo real, a representação deve atender aos seguintes critérios:
1. **Invariância de Posição**: Os Pokémon do banco de reservas devem ser representados de forma consistente (ordenados por status de vida ou processados via mecanismo de atenção/Transformer).
2. **Normalização Contínua**: Variáveis contínuas (HP, turnos restantes, boosts) são normalizadas no intervalo $[0, 1]$ ou $[-1, 1]$.
3. **Embeddings Categóricos**: Identificadores de espécies, itens, habilidades e golpes são mapeados para matrizes densas de embeddings (dimensão $32$ a $64$).
4. **Action Masking Completo**: Um vetor booleano $\mathbf{M} \in \{0, 1\}^{22}$ acompanha cada estado indicando exatamente quais ações são legais de acordo com o protocolo Showdown.

---

## 2. Estrutura do Vetor de Estado

O vetor de estado completo é composto por 4 blocos concatenados ou alimentados em submódulos separados da rede neural:

```
+---------------------------------------------------------------------------------------+
|                                TENSOR DE ESTADO DA BATALHA                            |
+---------------------------+---------------------------+---------------+---------------+
| 1. Pokémon Ativos         | 2. Banco de Reservas      | 3. Campo &    | 4. Crença     |
|    - Aliado Ativo         |    - Banco Aliado (5)     |    Ambiente   |    Oculta     |
|    - Inimigo Ativo        |    - Banco Inimigo (5)    |    - Clima    |    - Moves    |
|    (Tipos, Stats, HP,     |    (HP, Status, Revelado, |    - Terrenos |    - Itens    |
|     Boosts, Status, Moves)|     Tipos, Moves conhecidos) - Hazards    |    - Habilidade|
+---------------------------+---------------------------+---------------+---------------+
```

---

## 3. Especificação Detalhada por Bloco

### 3.1 Pokémon Ativo (Aliado e Oponente)
Cada Pokémon ativo é codificado com as seguintes características:

| Característica | Tipo | Dimensão | Descrição / Normalização |
|---|---|---|---|
| `Species ID` | Categórico | 1 | ID canônico para lookup na tabela de embeddings (ex: `dragapult` $\to 342$) |
| `Types` | One-Hot / Multi-Hot | 18 | Codificação binária dos 1 ou 2 tipos do Pokémon |
| `HP Atual` | Contínuo | 1 | $\text{HP}_{\text{atual}} / \text{HP}_{\text{máx}} \in [0.0, 1.0]$ |
| `Status` | One-Hot | 7 | `[None, Burn, Freeze, Paralysis, Poison, Toxic, Sleep]` |
| `Toxic Counter` | Discreto | 1 | Contador de turnos de veneno grave ($n / 15$) |
| `Sleep Counter` | Discreto | 1 | Turnos restantes de sono ($n / 3$) |
| `Stat Boosts` | Contínuo | 7 | Estágios de Atk, Def, SpA, SpD, Spe, Acc, Eva normalizados: $\text{boost} / 6.0 \in [-1.0, 1.0]$ |
| `Tera Type` | One-Hot | 19 | Tipo Terastal ativo ou planejado (incluindo tipo `Stellar`) |
| `Terastallized?` | Binário | 1 | Flag indicando se a mecânica já foi ativada |
| `Item Status` | Categórico/Binário | 2 | Flag de posse (`has_item`), `is_knocked_off` |
| `Volatile Statuses` | Multi-Hot | 12 | `[Substitute, Confusion, Taunt, Encore, Leech Seed, Magnet Rise, ...]` |
| `Moves (1 a 4)` | Categórico + Contínuo | $4 \times (1 + 1)$ | IDs de cada golpe + PP restante relativo ($\text{PP}_{\text{atual}} / \text{PP}_{\text{máx}}$) |

### 3.2 Banco de Reservas (Bench Pokémon)
Para cada um dos 5 Pokémon restantes de cada jogador:
- ID da Espécie (ou token `Unknown` se ainda não foi revelado no time adversário).
- HP relativo ($0.0$ caso esteja nocauteado / *fainted*).
- Condição de status principal.
- Multi-hot de golpes já revelados na partida (até 4 golpes).
- Flag de Pokémon vivo (`is_alive`: $0$ ou $1$).

### 3.3 Campo Global e Lados da Arena (Global Field State)
- **Clima**: One-Hot de 5 estados `[None, Sun, Rain, Sand, Snow]` + turnos restantes normalizados ($t / 8.0$).
- **Terreno**: One-Hot de 5 estados `[None, Electric, Grassy, Psychic, Misty]` + turnos restantes ($t / 8.0$).
- **Hazards Aliados e Inimigos**:
  - `Stealth Rock`: Binário ($0$ ou $1$).
  - `Spikes`: Contínuo ($\text{camadas} / 3.0$).
  - `Toxic Spikes`: Contínuo ($\text{camadas} / 2.0$).
  - `Sticky Web`: Binário ($0$ ou $1$).
- **Efeitos de Lado (Screens & Modifiers)**:
  - `Reflect`, `Light Screen`, `Aurora Veil`, `Tailwind`, `Trick Room` com seus respectivos contadores de turnos.
- **Turno da Partida**: Contador normalizado ($\min(\text{turn}, 60) / 60.0$).

---

## 4. O Espaço de Ações e Validação (Action Space)

O espaço discreto de tomada de decisão é unificado em **22 ações possíveis**:

```
Índices de Ação Discreta:
[0..3]   : Usar Move 1, 2, 3, 4 (Golpe Padrão)
[4..8]   : Trocar para Banco Slot 1, 2, 3, 4, 5
[9..12]  : Usar Move 1..4 com Terastallize ativado
[13..16] : Usar Move 1..4 com Mega Evolução
[17..20] : Usar Move 1..4 com Dynamax / Gigantamax
[21]     : Usar Revival Blessing / Ação Forçada de Luta (Struggle)
```

### Máscara de Ações (Action Mask):
A máscara $\mathbf{M} \in \{0, 1\}^{22}$ é computada antes de qualquer chamada de `softmax` a partir do payload de `|request|` recebido:
- Golpes sem PP restante têm máscara $0$.
- Movimentos bloqueados por *Taunt* (golpes de suporte), *Disable*, *Torment* ou *Choice Item* têm máscara $0$.
- Trocas para Pokémon nocauteados ou bloqueadas por efeitos de aprisionamento (*Shadow Tag*, *Arena Trap*, *Infestation*) têm máscara $0$.
- Se o Pokémon já gastou a mecânica Terastal ou Mega na partida, as ações $[9..20]$ têm máscara $0$.
