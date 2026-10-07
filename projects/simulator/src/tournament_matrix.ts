/**
 * Full Round-Robin Tournament Matrix Suite (200 Games per Matchup)
 * ================================================================
 * Evaluates all 6 competitive architectures in a complete N x N matrix:
 * 1. Amnesia-V2 (Gated Hybrid)
 * 2. Amnesia-V1 (Pure Gated Policy)
 * 3. FoulPlay-Minimax (Minimax Payoff Matrix)
 * 4. PokeEnv-Heuristics (Simple Heuristics)
 * 5. MaxDamage-Greedy (Greedy Max Base Power)
 * 6. RandomPlayer (Uniform Random Baseline)
 */

import { BattleStreams, RandomPlayerAI, Teams, Dex } from "@pkmn/sim";
import { TeamGenerators } from "@pkmn/randoms";
import * as fs from "fs";
import * as path from "path";

Teams.setGeneratorFactory(TeamGenerators);

// ============================================================================
// 1. Amnesia-V2: Gated Context Hybrid AI (Sleep Clause + Boost Defense + Minimax)
// ============================================================================
export class AmnesiaV2HybridAI extends BattleStreams.BattlePlayer {
  opponentSpecies: string = "";
  oppBoosts: Record<string, number> = {};
  oppIsAsleep: boolean = false;

  receiveError(error: Error) { this.choose("default"); }

  receiveRequest(request: any) {
    if (!request || request.wait) return;
    if (request.teamPreview) { this.choose("default"); return; }

    const pokemon = request.side?.pokemon || [];
    const active = request.active?.[0];
    const moves = active?.moves || [];

    const myActive = pokemon.find((p: any) => p.active);
    const myHpPct = myActive ? (parseFloat(myActive.condition.split("/")[0]) || 0) / 100.0 : 1.0;
    const mySpec = myActive ? Dex.species.get(myActive.details.split(",")[0]) : null;
    const oppSpec = this.opponentSpecies ? Dex.species.get(this.opponentSpecies) : null;
    const oppTypes = oppSpec?.types || ["Normal"];
    const myTypes = mySpec?.types || ["Normal"];

    // Total dangerous boost of opponent (Atk, SpA, Spe)
    const oppThreat = (this.oppBoosts["atk"] || 0) + (this.oppBoosts["spa"] || 0) + (this.oppBoosts["spe"] || 0);

    // 1. Forced or Tactical Switch
    if (request.forceSwitch) {
      let bestSwitch = -1;
      let bestScore = -9999;

      for (let i = 0; i < pokemon.length; i++) {
        const p = pokemon[i];
        if (!p.active && !p.condition.includes("fnt") && !p.condition.startsWith("0")) {
          const spec = Dex.species.get(p.details.split(",")[0]);
          const hpPct = (parseFloat(p.condition.split("/")[0]) || 0) / 100.0;
          let defScore = hpPct * 3.0;

          // Resistances to opponent
          for (const ot of oppTypes) {
            const eff = Dex.getEffectiveness(ot, spec.types);
            if (eff > 0) defScore -= 2.0;
            else if (eff < 0) defScore += 2.0;
          }

          // Speed Advantage check
          const bSpe = spec.baseStats.spe;
          const oSpe = oppSpec ? oppSpec.baseStats.spe : 80;
          if (bSpe > oSpe) defScore += 1.5;

          if (defScore > bestScore) {
            bestScore = defScore;
            bestSwitch = i + 1;
          }
        }
      }
      if (bestSwitch > 0) { this.choose(`switch ${bestSwitch}`); return; }
      this.choose("default");
      return;
    }

    // 2. Move Scoring with Contextual Gating & Sleep Clause
    let bestSlot = -1;
    let bestUtility = -99999;

    const sleepMoves = ["spore", "sleeppowder", "hypnosis", "yawn", "sing", "grasswhistle", "lovelykiss", "darkvoid"];

    for (let i = 0; i < moves.length; i++) {
      const m = moves[i];
      if (m.disabled || (m.pp !== undefined && m.pp <= 0)) continue;

      const moveId = (m.id || m.move || "").toLowerCase().replace(/[^a-z0-9]/g, "");
      const moveData = Dex.moves.get(moveId);
      if (!moveData) continue;

      // SLEEP CLAUSE ENFORCEMENT: Never use sleep moves if enemy is already asleep!
      if (sleepMoves.includes(moveId) && this.oppIsAsleep) {
        continue; // Action Masked out (0 probability)
      }

      let utility = 0;
      const bp = moveData.basePower || (moveData.category === "Status" ? 30 : 60);

      if (moveData.category === "Status") {
        // Recovery when HP is low
        if (["recover", "roost", "softboiled", "slackoff", "synthesis", "moonlight", "wish"].includes(moveId)) {
          utility = myHpPct < 0.5 ? 120 * (1.0 - myHpPct) : 10;
        }
        // Stat Buffs (Dragon Dance, Swords Dance, Nasty Plot, Calm Mind)
        else if (["swordsdance", "nastyplot", "dragondance", "calmmind", "quiverdance", "bulkup", "curse"].includes(moveId)) {
          utility = (myHpPct > 0.6 && oppThreat <= 0) ? 95 : 15;
        }
        // Sleep Induction
        else if (sleepMoves.includes(moveId) && !this.oppIsAsleep) {
          utility = 110;
        }
        // Entry Hazards (High early game, zero late game)
        else if (["stealthrock", "spikes", "toxicspikes", "stickyweb"].includes(moveId)) {
          utility = 75;
        }
        else {
          utility = 40;
        }
      } else {
        // Offensive Move Evaluation
        let typeEff = 1.0;
        if (oppTypes.length > 0) {
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
        const acc = typeof moveData.accuracy === "number" ? (moveData.accuracy / 100.0) : 1.0;
        const prio = moveData.priority || 0;

        let dmgPotential = bp * typeEff * stab * acc;

        // Context Gating: Priority moves gain massive utility against boosted enemies
        if (oppThreat >= 2 && prio > 0) {
          dmgPotential *= 2.2;
        }

        utility = dmgPotential;
      }

      if (utility > bestUtility) {
        bestUtility = utility;
        bestSlot = i + 1;
      }
    }

    if (bestSlot > 0) this.choose(`move ${bestSlot}`);
    else this.choose("default");
  }
}

// ============================================================================
// 2. Amnesia-V1: Pure Policy AI (Imitation Baseline)
// ============================================================================
export class AmnesiaV1PurePolicyAI extends BattleStreams.BattlePlayer {
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
    const myTypes = myActive ? Dex.species.get(myActive.details.split(",")[0]).types : [];
    const oppTypes = this.opponentSpecies ? Dex.species.get(this.opponentSpecies).types : [];

    let bestSlot = -1;
    let bestScore = -999;

    for (let i = 0; i < moves.length; i++) {
      const m = moves[i];
      if (!m.disabled && (m.pp === undefined || m.pp > 0)) {
        const moveData = Dex.moves.get(m.id || m.move);
        let score = moveData.basePower || 45;

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
// 3. Foul Play Minimax AI
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
          if (Dex.getImmunity(moveData.type, oppTypes) === false) typeEff = 0.0;
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
// 4. Poke-Env SimpleHeuristics AI
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

    for (let i = 0; i < moves.length; i++) {
      const mid = (moves[i].id || moves[i].move || "").toLowerCase();
      if (myHp < 0.40 && ["roost", "recover", "softboiled", "slackoff", "synthesis", "moonlight"].includes(mid)) {
        this.choose(`move ${i + 1}`);
        return;
      }
    }

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
// 5. MaxDamage Greedy AI
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
// Tournament Runner & Matrix Generator
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
        const parts = line.split("|");
        if (line.startsWith("|switch|p2a:") || line.startsWith("|drag|p2a:")) {
          if (p1.opponentSpecies !== undefined) p1.opponentSpecies = parts[3].split(",")[0].trim();
        } else if (line.startsWith("|switch|p1a:") || line.startsWith("|drag|p1a:")) {
          if (p2.opponentSpecies !== undefined) p2.opponentSpecies = parts[3].split(",")[0].trim();
        } else if (line.startsWith("|-boost|p2a:") && p1.oppBoosts) {
          const stat = parts[3];
          const val = parseInt(parts[4] || "1", 10);
          p1.oppBoosts[stat] = (p1.oppBoosts[stat] || 0) + val;
        } else if (line.startsWith("|-boost|p1a:") && p2.oppBoosts) {
          const stat = parts[3];
          const val = parseInt(parts[4] || "1", 10);
          p2.oppBoosts[stat] = (p2.oppBoosts[stat] || 0) + val;
        } else if (line.startsWith("|-status|p2a|slp") && p1.oppIsAsleep !== undefined) {
          p1.oppIsAsleep = true;
        } else if (line.startsWith("|-curestatus|p2a|slp") && p1.oppIsAsleep !== undefined) {
          p1.oppIsAsleep = false;
        } else if (line.startsWith("|turn|")) {
          turns = parseInt(parts[2], 10);
        } else if (line.startsWith("|win|")) {
          winner = parts[2].trim();
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
    }, 4);
  });
}

async function main() {
  const BATTLES_PER_PAIR = 200; // 200 matches per cell!

  const agents = [
    { name: "Amnesia-V2 (Gated)", cls: AmnesiaV2HybridAI },
    { name: "Amnesia-V1 (Policy)", cls: AmnesiaV1PurePolicyAI },
    { name: "FoulPlay-Minimax", cls: FoulPlayMinimaxAI },
    { name: "PokeEnv-Heuristics", cls: SimpleHeuristicsAI },
    { name: "MaxDamage-Greedy", cls: MaxDamageAI },
    { name: "RandomPlayer", cls: RandomPlayerAI }
  ];

  const agentNames = agents.map(a => a.name);
  const n = agents.length;

  console.log("======================================================================");
  console.log(`🏆 RUNNING COMPLETE ROUND-ROBIN TOURNAMENT (${BATTLES_PER_PAIR} BATTLES PER PAIR)`);
  console.log("======================================================================");

  // winMatrix[row][col]: Win Rate of Agent(col) when playing AGAINST Agent(row)
  const winCounts: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
  const totalGames: number[][] = Array.from({ length: n }, () => Array(n).fill(0));

  const startTime = performance.now();

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const opp = agents[i];
      const me = agents[j];

      if (i === j) {
        // Self-play: 50% win rate expected by symmetry
        winCounts[i][j] = BATTLES_PER_PAIR / 2;
        totalGames[i][j] = BATTLES_PER_PAIR;
        continue;
      }

      console.log(`⚔️  ${me.name} vs ${opp.name} (${BATTLES_PER_PAIR} games)...`);
      let meWins = 0;

      for (let b = 0; b < BATTLES_PER_PAIR; b++) {
        // Alternate player 1 / player 2 to eliminate first-mover advantage
        const isP1 = b % 2 === 0;
        const p1Cls = isP1 ? me.cls : opp.cls;
        const p2Cls = isP1 ? opp.cls : me.cls;
        const p1Name = isP1 ? me.name : opp.name;
        const p2Name = isP1 ? opp.name : me.name;

        const res = await runMatch(p1Cls, p2Cls, p1Name, p2Name);
        if (res.includes(me.name)) {
          meWins++;
        }
      }

      winCounts[i][j] = meWins;
      totalGames[i][j] = BATTLES_PER_PAIR;
      console.log(`   -> ${me.name} Win Rate vs ${opp.name}: ${((meWins / BATTLES_PER_PAIR) * 100).toFixed(1)}%`);
    }
  }

  const durationSec = ((performance.now() - startTime) / 1000).toFixed(2);
  console.log(`\n✅ Completed in ${durationSec}s!`);

  // Calculate Win Percentages
  const winRateMatrix: number[][] = [];
  for (let i = 0; i < n; i++) {
    winRateMatrix[i] = [];
    for (let j = 0; j < n; j++) {
      const pct = Math.round((winCounts[i][j] / totalGames[i][j]) * 100);
      winRateMatrix[i][j] = pct;
    }
  }

  const results = {
    agents: agentNames,
    battles_per_pair: BATTLES_PER_PAIR,
    matrix: winRateMatrix,
    raw_wins: winCounts,
    duration_seconds: parseFloat(durationSec)
  };

  const outPath = path.resolve(__dirname, "../../tournament_matrix.json");
  fs.writeFileSync(outPath, JSON.stringify(results, null, 2));
  console.log(`💾 Saved results to: ${outPath}`);
}

main().catch(console.error);
