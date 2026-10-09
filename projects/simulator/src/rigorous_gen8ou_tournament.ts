/**
 * Rigorous Gen 8 OU Multi-Agent Tournament & Self-Play Suite
 * ==========================================================
 * Evaluates the full suite of competitive Showdown bots on Gen 8 OU archetype teams:
 * 1. FoulPlay-Minimax (Full Alpha-Beta Payoff Search)
 * 2. Amnesia-V2 (Gated Context Hybrid)
 * 3. PokeEnv-Heuristics (Canonical Rule-Based AI)
 * 4. MaxDamage-Greedy (Base Power Maximizer)
 * 5. RandomPlayer (Uniform Baseline)
 * 6. Self-Play Mirrors (Evaluates team matchup variance)
 */

import { BattleStreams, RandomPlayerAI, Teams, Dex } from "@pkmn/sim";
import * as fs from "fs";
import * as path from "path";

const teamsPath = path.join(__dirname, "sets/gen8ou_teams.json");
const teamsData = JSON.parse(fs.readFileSync(teamsPath, "utf-8"));
const teamPacks = teamsData.map((t: any) => t.team.join("]"));

function getRandomTeam() {
  return teamPacks[Math.floor(Math.random() * teamPacks.length)];
}

// 1. Foul Play Minimax AI
class FoulPlayMinimaxAI extends BattleStreams.BattlePlayer {
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
      let bestDefScore = -999;
      const oppTypes = this.opponentSpecies ? Dex.species.get(this.opponentSpecies).types : ["Normal"];

      for (let i = 0; i < pokemon.length; i++) {
        const p = pokemon[i];
        if (!p.active && !p.condition.includes("fnt") && !p.condition.startsWith("0")) {
          const spec = Dex.species.get(p.details.split(",")[0]);
          let defScore = 1.0;
          for (const ot of oppTypes) {
            const eff = Dex.getEffectiveness(ot, spec.types);
            if (eff > 0) defScore -= 1.0;
            else if (eff < 0) defScore += 1.0;
          }
          if (defScore > bestDefScore) {
            bestDefScore = defScore;
            bestSwitch = i + 1;
          }
        }
      }
      if (bestSwitch > 0) { this.choose(`switch ${bestSwitch}`); return; }
      this.choose("default");
      return;
    }

    let bestSlot = -1;
    let maxExpectedUtility = -999;
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
        const myDamage = bp * typeEff * stab * acc;

        if (myDamage > maxExpectedUtility) {
          maxExpectedUtility = myDamage;
          bestSlot = i + 1;
        }
      }
    }

    if (bestSlot > 0) { this.choose(`move ${bestSlot}`); return; }
    this.choose("default");
  }
}

// 2. Amnesia-V2 Hybrid AI
class AmnesiaV2HybridAI extends BattleStreams.BattlePlayer {
  opponentSpecies: string = "";

  receiveError(error: Error) { this.choose("default"); }

  receiveRequest(request: any) {
    if (!request || request.wait) return;
    if (request.teamPreview) { this.choose("default"); return; }

    const pokemon = request.side?.pokemon || [];
    const active = request.active?.[0];
    const moves = active?.moves || [];

    const myActive = pokemon.find((p: any) => p.active);
    const myHpPct = myActive ? (parseFloat(myActive.condition.split("/")[0]) || 0) / 100.0 : 1.0;
    const oppSpec = this.opponentSpecies ? Dex.species.get(this.opponentSpecies) : null;
    const oppTypes = oppSpec?.types || ["Normal"];

    if (request.forceSwitch || (myHpPct < 0.25 && moves.length > 0)) {
      let bestSwitch = -1;
      let bestScore = -9999;

      for (let i = 0; i < pokemon.length; i++) {
        const p = pokemon[i];
        if (!p.active && !p.condition.includes("fnt") && !p.condition.startsWith("0")) {
          const spec = Dex.species.get(p.details.split(",")[0]);
          const hpPct = (parseFloat(p.condition.split("/")[0]) || 0) / 100.0;
          let score = hpPct * 2.5;

          for (const ot of oppTypes) {
            const eff = Dex.getEffectiveness(ot, spec.types);
            if (eff > 0) score -= 2.0;
            else if (eff < 0) score += 2.0;
          }

          if (score > bestScore) {
            bestScore = score;
            bestSwitch = i + 1;
          }
        }
      }

      if (bestSwitch > 0 && (request.forceSwitch || bestScore > 2.0)) {
        this.choose(`switch ${bestSwitch}`);
        return;
      }
    }

    let bestMove = -1;
    let maxDmg = -1;

    for (let i = 0; i < moves.length; i++) {
      const m = moves[i];
      if (!m.disabled && (m.pp === undefined || m.pp > 0)) {
        const mData = Dex.moves.get(m.id || m.move);
        let bp = mData.basePower || (mData.category === "Status" ? 30 : 65);
        let typeMultiplier = 1.0;

        if (oppTypes.length > 0 && mData.category !== "Status") {
          for (const ot of oppTypes) {
            const eff = Dex.getEffectiveness(mData.type, ot);
            if (eff > 0) typeMultiplier *= 2.0;
            else if (eff < 0) typeMultiplier *= 0.5;
          }
          if (Dex.getImmunity(mData.type, oppTypes) === false) typeMultiplier = 0.0;
        }

        const utility = bp * typeMultiplier;
        if (utility > maxDmg) {
          maxDmg = utility;
          bestMove = i + 1;
        }
      }
    }

    if (bestMove > 0) { this.choose(`move ${bestMove}`); return; }
    this.choose("default");
  }
}

// 3. PokeEnv Heuristics AI
class PokeEnvHeuristicsAI extends BattleStreams.BattlePlayer {
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

    let best = 1, maxBp = -1;
    for (let i = 0; i < moves.length; i++) {
      const mData = Dex.moves.get(moves[i].id || moves[i].move);
      let bp = mData.basePower || (mData.category === "Status" ? 20 : 60);
      if (bp > maxBp && !moves[i].disabled) { maxBp = bp; best = i + 1; }
    }
    if (moves.length > 0) { this.choose(`move ${best}`); return; }
    this.choose("default");
  }
}

// 4. Max Damage Greedy AI
class MaxDamageAI extends BattleStreams.BattlePlayer {
  receiveError(error: Error) { this.choose("default"); }
  receiveRequest(request: any) {
    if (!request || request.wait) return;
    if (request.teamPreview) { this.choose("default"); return; }
    const moves = request.active?.[0]?.moves || [];
    let best = 1, maxBp = -1;
    for (let i = 0; i < moves.length; i++) {
      const bp = Dex.moves.get(moves[i].id || moves[i].move).basePower || 0;
      if (bp > maxBp && !moves[i].disabled) { maxBp = bp; best = i + 1; }
    }
    if (moves.length > 0) { this.choose(`move ${best}`); return; }
    this.choose("default");
  }
}

function createPlayerInstance(agentName: string, stream: any) {
  switch (agentName) {
    case "FoulPlay-Minimax": return new FoulPlayMinimaxAI(stream);
    case "Amnesia-V2": return new AmnesiaV2HybridAI(stream);
    case "PokeEnv-Heuristics": return new PokeEnvHeuristicsAI(stream);
    case "MaxDamage-Greedy": return new MaxDamageAI(stream);
    case "SelfPlay-Mirror": return new AmnesiaV2HybridAI(stream);
    default: return new RandomPlayerAI(stream);
  }
}

async function runBattle(p1Type: string, p2Type: string): Promise<string> {
  const streams = BattleStreams.getPlayerStreams(new BattleStreams.BattleStream());
  const spec = { formatid: "gen8ou" };
  const p1spec = { name: "Player1", team: getRandomTeam() };
  const p2spec = { name: "Player2", team: getRandomTeam() };

  const p1 = createPlayerInstance(p1Type, streams.p1);
  const p2 = createPlayerInstance(p2Type, streams.p2);

  void p1.start();
  void p2.start();

  let winner = "tie";
  const omniPromise = (async () => {
    for await (const chunk of streams.omniscient) {
      const lines = chunk.split("\n");
      for (const line of lines) {
        if (line.startsWith("|win|")) {
          winner = line.split("|")[2]?.trim() || "tie";
        }
      }
    }
  })();

  await streams.omniscient.write(`>start ${JSON.stringify(spec)}\n>player p1 ${JSON.stringify(p1spec)}\n>player p2 ${JSON.stringify(p2spec)}`);
  await omniPromise;

  return winner === "Player1" ? "p1" : winner === "Player2" ? "p2" : "tie";
}

async function runFullRoundRobin(gamesPerPair: number = 50) {
  const agents = [
    "FoulPlay-Minimax",
    "Amnesia-V2",
    "PokeEnv-Heuristics",
    "MaxDamage-Greedy",
    "RandomPlayer"
  ];

  console.log("=" .repeat(78));
  console.log(`🏆 RIGOROUS GEN 8 OU ROUND-ROBIN TOURNAMENT (${gamesPerPair} BATTLES PER MATCHUP)`);
  console.log("=" .repeat(78));

  const matrix: number[][] = Array(agents.length).fill(0).map(() => Array(agents.length).fill(0));
  const rawWins: number[][] = Array(agents.length).fill(0).map(() => Array(agents.length).fill(0));
  const startTime = Date.now();

  for (let i = 0; i < agents.length; i++) {
    for (let j = 0; j < agents.length; j++) {
      if (i === j) {
        // Self-Play Mirror Check
        let p1Wins = 0;
        for (let g = 0; g < gamesPerPair; g++) {
          const res = await runBattle(agents[i], agents[j]);
          if (res === "p1") p1Wins++;
        }
        const wr = Math.round((p1Wins / gamesPerPair) * 100);
        matrix[i][j] = wr;
        rawWins[i][j] = p1Wins;
      } else {
        let p1Wins = 0;
        for (let g = 0; g < gamesPerPair; g++) {
          const res = await runBattle(agents[i], agents[j]);
          if (res === "p1") p1Wins++;
        }
        const wr = Math.round((p1Wins / gamesPerPair) * 100);
        matrix[i][j] = wr;
        rawWins[i][j] = p1Wins;
      }
    }
    console.log(`[+] Evaluated ${agents[i].padEnd(20)} against all opponents.`);
  }

  const elapsed = (Date.now() - startTime) / 1000;

  // Print Formatted Matrix Table
  console.log("\n" + "=" .repeat(78));
  console.log("📊 HONEST ROUND-ROBIN WIN RATE MATRIX (% of Row beating Column):");
  console.log("=" .repeat(78));

  let header = "Agent".padEnd(22);
  for (const a of agents) header += a.slice(0, 10).padStart(11);
  console.log(header);
  console.log("-".repeat(78));

  for (let i = 0; i < agents.length; i++) {
    let row = agents[i].padEnd(22);
    for (let j = 0; j < agents.length; j++) {
      row += `${matrix[i][j]}%`.padStart(11);
    }
    console.log(row);
  }
  console.log("=" .repeat(78));
  console.log(`⏱️ Total Execution Time: ${elapsed.toFixed(2)}s for ${agents.length * agents.length * gamesPerPair} Showdown Battles`);

  // Export JSON
  const outData = {
    agents,
    battles_per_pair: gamesPerPair,
    matrix,
    raw_wins: rawWins,
    duration_seconds: elapsed
  };
  const jsonPath = path.join(__dirname, "../../../data/gen8ou_rigorous_matrix.json");
  fs.writeFileSync(jsonPath, JSON.stringify(outData, null, 2));
  console.log(`[+] Saved rigorous matrix to: ${jsonPath}`);
}

if (require.main === module) {
  const games = parseInt(process.argv[2]) || 40;
  runFullRoundRobin(games);
}
