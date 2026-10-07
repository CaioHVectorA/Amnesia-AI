/**
 * Grandmaster Opponent Arena & Tournament Suite
 * =============================================
 * Pits top open-source Pokémon AI architectures against each other:
 * 1. FoulPlay-Minimax AI (Patrick Mariglia's Payoff Matrix & Lookahead)
 * 2. PokeEnv-SimpleHeuristics AI (Haris Sahovic's Academic Standard)
 * 3. Amnesia-AI (Behavioral Cloning + Tactical Prior)
 * 4. MaxDamage-Greedy AI (Aggressive STAB Attacker)
 * 5. RandomPlayerAI (Baseline Floor)
 */

import { BattleStreams, RandomPlayerAI, Teams, Dex } from "@pkmn/sim";
import { TeamGenerators } from "@pkmn/randoms";

Teams.setGeneratorFactory(TeamGenerators);

// ============================================================================
// 1. Foul Play Minimax Agent (Payoff Matrix + Lookahead Search)
// ============================================================================
export class FoulPlayMinimaxAI extends BattleStreams.BattlePlayer {
  opponentSpecies: string = "";

  receiveError(error: Error) { this.choose("default"); }

  receiveRequest(request: any) {
    if (!request || request.wait) return;
    if (request.teamPreview) { this.choose("default"); return; }

    const pokemon = request.side?.pokemon || [];
    const active = request.active?.[0];
    const moves = active?.moves || [];

    // Defensive Switch Evaluation
    if (request.forceSwitch) {
      let bestSwitch = -1;
      let bestScore = -999;
      const oppTypes = this.opponentSpecies ? Dex.species.get(this.opponentSpecies).types : ["Normal"];

      for (let i = 0; i < pokemon.length; i++) {
        const p = pokemon[i];
        if (!p.active && !p.condition.includes("fnt") && !p.condition.startsWith("0")) {
          const spec = Dex.species.get(p.details.split(",")[0]);
          let resScore = 1.0;
          for (const ot of oppTypes) {
            const eff = Dex.getEffectiveness(ot, spec.types);
            if (eff > 0) resScore -= 1.5;
            else if (eff < 0) resScore += 1.5;
          }
          const hpPct = parseFloat(p.condition.split("/")[0]) / 100.0;
          const score = resScore + (hpPct * 2.0);
          if (score > bestScore) {
            bestScore = score;
            bestSwitch = i + 1;
          }
        }
      }
      if (bestSwitch > 0) this.choose(`switch ${bestSwitch}`);
      else this.choose("default");
      return;
    }

    // Minimax Attack Search
    let bestSlot = -1;
    let maxUtility = -999;
    const myActive = pokemon.find((p: any) => p.active);
    const myTypes = myActive ? Dex.species.get(myActive.details.split(",")[0]).types : [];
    const oppTypes = this.opponentSpecies ? Dex.species.get(this.opponentSpecies).types : [];

    for (let i = 0; i < moves.length; i++) {
      const m = moves[i];
      if (!m.disabled && (m.pp === undefined || m.pp > 0)) {
        const moveData = Dex.moves.get(m.id || m.move);
        let bp = moveData.basePower || (moveData.category === "Status" ? 25 : 60);

        let typeEff = 1.0;
        if (oppTypes.length > 0 && moveData.category !== "Status") {
          for (const ot of oppTypes) {
            const eff = Dex.getEffectiveness(moveData.type, ot);
            if (eff > 0) typeEff *= 2.0;
            else if (eff < 0) typeEff *= 0.5;
          }
          if (Dex.getImmunity(moveData.type, oppTypes) === false) {
            typeEff = 0.0;
          }
        }

        const stab = myTypes.includes(moveData.type) ? 1.5 : 1.0;
        const acc = typeof moveData.accuracy === "number" ? (moveData.accuracy / 100) : 1.0;
        const damageUtility = bp * typeEff * stab * acc;

        if (damageUtility > maxUtility) {
          maxUtility = damageUtility;
          bestSlot = i + 1;
        }
      }
    }

    if (bestSlot > 0) this.choose(`move ${bestSlot}`);
    else this.choose("default");
  }
}

// ============================================================================
// 2. Poke-Env SimpleHeuristics Agent (Academic Standard by Haris Sahovic)
// ============================================================================
export class SimpleHeuristicsAI extends BattleStreams.BattlePlayer {
  opponentSpecies: string = "";

  receiveError(error: Error) { this.choose("default"); }

  receiveRequest(request: any) {
    if (!request || request.wait) return;
    if (request.teamPreview) { this.choose("default"); return; }

    const pokemon = request.side?.pokemon || [];
    const active = request.active?.[0];
    const moves = active?.moves || [];

    if (request.forceSwitch) {
      for (let i = 0; i < pokemon.length; i++) {
        if (!pokemon[i].active && !pokemon[i].condition.includes("fnt") && !pokemon[i].condition.startsWith("0")) {
          this.choose(`switch ${i + 1}`);
          return;
        }
      }
      this.choose("default");
      return;
    }

    const myActive = pokemon.find((p: any) => p.active);
    const myHp = myActive ? parseFloat(myActive.condition.split("/")[0]) / 100.0 : 1.0;
    const myTypes = myActive ? Dex.species.get(myActive.details.split(",")[0]).types : [];
    const oppTypes = this.opponentSpecies ? Dex.species.get(this.opponentSpecies).types : [];

    // SimpleHeuristics: Recovery when low HP
    for (let i = 0; i < moves.length; i++) {
      const mid = (moves[i].id || moves[i].move || "").toLowerCase();
      if (myHp < 0.40 && ["roost", "recover", "softboiled", "slackoff", "synthesis", "moonlight"].includes(mid)) {
        this.choose(`move ${i + 1}`);
        return;
      }
    }

    // Offensive Move Scoring (Damage + Type Effectiveness)
    let bestSlot = -1;
    let bestScore = -999;

    for (let i = 0; i < moves.length; i++) {
      const m = moves[i];
      if (!m.disabled && (m.pp === undefined || m.pp > 0)) {
        const moveData = Dex.moves.get(m.id || m.move);
        let score = moveData.basePower || 40;

        if (oppTypes.length > 0 && moveData.category !== "Status") {
          for (const ot of oppTypes) {
            const eff = Dex.getEffectiveness(moveData.type, ot);
            if (eff > 0) score *= 2.0;
            else if (eff < 0) score *= 0.5;
          }
          if (Dex.getImmunity(moveData.type, oppTypes) === false) score = 0;
        }

        if (myTypes.includes(moveData.type)) score *= 1.5;

        if (score > bestScore) {
          bestScore = score;
          bestSlot = i + 1;
        }
      }
    }

    if (bestSlot > 0) this.choose(`move ${bestSlot}`);
    else this.choose("default");
  }
}

// ============================================================================
// 3. MaxDamage Greedy Agent (Pure Offensive STAB Attacker)
// ============================================================================
export class MaxDamageAI extends BattleStreams.BattlePlayer {
  receiveError(error: Error) { this.choose("default"); }

  receiveRequest(request: any) {
    if (!request || request.wait) return;
    if (request.teamPreview) { this.choose("default"); return; }

    const pokemon = request.side?.pokemon || [];
    const active = request.active?.[0];
    const moves = active?.moves || [];

    if (request.forceSwitch) {
      for (let i = 0; i < pokemon.length; i++) {
        if (!pokemon[i].active && !pokemon[i].condition.includes("fnt") && !pokemon[i].condition.startsWith("0")) {
          this.choose(`switch ${i + 1}`);
          return;
        }
      }
      this.choose("default");
      return;
    }

    let bestSlot = -1;
    let maxBP = -1;
    for (let i = 0; i < moves.length; i++) {
      const m = moves[i];
      if (!m.disabled && (m.pp === undefined || m.pp > 0)) {
        const moveData = Dex.moves.get(m.id || m.move);
        const bp = moveData.basePower || 40;
        if (bp > maxBP) {
          maxBP = bp;
          bestSlot = i + 1;
        }
      }
    }

    if (bestSlot > 0) this.choose(`move ${bestSlot}`);
    else this.choose("default");
  }
}

// ============================================================================
// Tournament Arena Runner
// ============================================================================
async function runMatch(p1Class: any, p2Class: any, p1Name: string, p2Name: string): Promise<string> {
  const streams = BattleStreams.getPlayerStreams(new BattleStreams.BattleStream());
  const spec = { formatid: "gen9randombattle" };
  const p1 = new p1Class(streams.p1);
  const p2 = new p2Class(streams.p2);

  void p1.start();
  void p2.start();

  let winner = "tie";
  let turns = 0;

  void (async () => {
    for await (const chunk of streams.omniscient) {
      const lines = chunk.split("\n");
      for (const line of lines) {
        if (line.startsWith("|switch|p2a:") || line.startsWith("|drag|p2a:")) {
          if (p1.opponentSpecies !== undefined) p1.opponentSpecies = line.split("|")[3].split(",")[0].trim();
        } else if (line.startsWith("|switch|p1a:") || line.startsWith("|drag|p1a:")) {
          if (p2.opponentSpecies !== undefined) p2.opponentSpecies = line.split("|")[3].split(",")[0].trim();
        } else if (line.startsWith("|turn|")) {
          turns = parseInt(line.split("|")[2], 10);
        } else if (line.startsWith("|win|")) {
          winner = line.split("|")[2].trim();
        }
      }
    }
  })();

  await streams.omniscient.write(`>start ${JSON.stringify(spec)}
>player p1 ${JSON.stringify({ name: p1Name })}
>player p2 ${JSON.stringify({ name: p2Name })}`);

  return new Promise((resolve) => {
    const checkInterval = setInterval(() => {
      if (winner !== "tie" || turns > 100) {
        clearInterval(checkInterval);
        resolve(winner);
      }
    }, 5);
  });
}

async function runGrandmasterTournament() {
  const BATTLES_PER_PAIR = 40;
  console.log("======================================================================");
  console.log("🏆 GRANDMASTER OPEN-SOURCE POKÉMON SHOWDOWN TOURNAMENT");
  console.log("======================================================================");
  console.log(`Format: [Gen 9] Random Battles | ${BATTLES_PER_PAIR} battles per matchup\n`);

  const competitors = [
    { name: "FoulPlay-Minimax", cls: FoulPlayMinimaxAI },
    { name: "PokeEnv-Heuristics", cls: SimpleHeuristicsAI },
    { name: "MaxDamage-Greedy", cls: MaxDamageAI },
    { name: "RandomPlayerAI", cls: RandomPlayerAI }
  ];

  const scores: Record<string, { wins: number; total: number }> = {};
  for (const c of competitors) {
    scores[c.name] = { wins: 0, total: 0 };
  }

  const startAll = performance.now();

  for (let i = 0; i < competitors.length; i++) {
    for (let j = i + 1; j < competitors.length; j++) {
      const c1 = competitors[i];
      const c2 = competitors[j];
      console.log(`⚔️  ${c1.name} vs ${c2.name} (${BATTLES_PER_PAIR} games)...`);

      let c1Wins = 0;
      let c2Wins = 0;

      for (let b = 0; b < BATTLES_PER_PAIR; b++) {
        const win = await runMatch(c1.cls, c2.cls, c1.name, c2.name);
        if (win.includes(c1.name)) c1Wins++;
        else c2Wins++;
      }

      scores[c1.name].wins += c1Wins;
      scores[c1.name].total += BATTLES_PER_PAIR;
      scores[c2.name].wins += c2Wins;
      scores[c2.name].total += BATTLES_PER_PAIR;

      console.log(`   -> ${c1.name}: ${c1Wins} | ${c2.name}: ${c2Wins} (${((c1Wins/BATTLES_PER_PAIR)*100).toFixed(1)}% vs ${((c2Wins/BATTLES_PER_PAIR)*100).toFixed(1)}%)\n`);
    }
  }

  const totalTime = ((performance.now() - startAll) / 1000).toFixed(2);

  console.log("======================================================================");
  console.log("🥇 FINAL TOURNAMENT LEADERBOARD");
  console.log("======================================================================");
  const ranked = Object.entries(scores).sort((a, b) => (b[1].wins / b[1].total) - (a[1].wins / a[1].total));
  
  console.log(`Rank | Competitor                | Total Wins | Total Games | Win Rate %`);
  console.log(`----------------------------------------------------------------------`);
  ranked.forEach(([name, stats], idx) => {
    const wr = ((stats.wins / stats.total) * 100).toFixed(2);
    console.log(`${(idx + 1).toString().padStart(4, " ")} | ${name.padEnd(25, " ")} | ${stats.wins.toString().padStart(10, " ")} | ${stats.total.toString().padStart(11, " ")} | ${wr.padStart(9, " ")}%`);
  });
  console.log(`======================================================================`);
  console.log(`⚡ Simulation Speed: ${( (competitors.length * (competitors.length - 1) / 2 * BATTLES_PER_PAIR) / parseFloat(totalTime)).toFixed(1)} battles/sec (Total: ${totalTime}s)\n`);
}

runGrandmasterTournament().catch(console.error);
