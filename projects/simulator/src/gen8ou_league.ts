/**
 * Gen 8 OU High-Speed Headless League & Transition Generator
 * ==========================================================
 * Powered by @pkmn/sim and Bun.
 * Evaluates agents in Gen 8 OU standard play across a pool of competitive archetype teams.
 */

import { BattleStreams, RandomPlayerAI, Teams, Dex } from "@pkmn/sim";
import * as fs from "fs";
import * as path from "path";

// Load Gen 8 OU Curated Teams
const teamsPath = path.join(__dirname, "sets/gen8ou_teams.json");
const teamsData = JSON.parse(fs.readFileSync(teamsPath, "utf-8"));
const teamPacks = teamsData.map((t: any) => t.team.join("]"));

function getRandomTeam() {
  return teamPacks[Math.floor(Math.random() * teamPacks.length)];
}

// 1. FoulPlay Minimax Agent (Gen 8 OU Aware)
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
      const oppTypes = this.opponentSpecies ? Dex.species.get(this.opponentSpecies).types : [];

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

// 2. Amnesia-V2 Gated Hybrid AI
class AmnesiaV2Gen8AI extends BattleStreams.BattlePlayer {
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

    // Gated switch logic
    if (request.forceSwitch || (myHpPct < 0.30 && moves.length > 0)) {
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

    // Move Selection with Type Synergy & High Power Priority
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

// 3. Greedy Max Damage Baseline
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

async function runGen8OUBattle(p1Type: string, p2Type: string): Promise<string> {
  const streams = BattleStreams.getPlayerStreams(new BattleStreams.BattleStream());
  const spec = { formatid: "gen8ou" };
  const p1spec = { name: "Agent_P1", team: getRandomTeam() };
  const p2spec = { name: "Agent_P2", team: getRandomTeam() };

  function createPlayer(type: string, stream: any) {
    switch (type) {
      case "Amnesia-V2": return new AmnesiaV2Gen8AI(stream);
      case "FoulPlay-Minimax": return new FoulPlayMinimaxAI(stream);
      case "MaxDamage": return new MaxDamageAI(stream);
      default: return new RandomPlayerAI(stream);
    }
  }

  const p1 = createPlayer(p1Type, streams.p1);
  const p2 = createPlayer(p2Type, streams.p2);

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

  return winner === "Agent_P1" ? "p1" : winner === "Agent_P2" ? "p2" : "tie";
}

async function runGen8OULadderBenchmark(numGames: number = 100) {
  console.log("=" .repeat(70));
  console.log(`⚔️ GEN 8 OU HEADLESS TOURNAMENT BENCHMARK (${numGames} Games/Pair)`);
  console.log("=" .repeat(70));

  const matchups = [
    { p1: "Amnesia-V2", p2: "MaxDamage" },
    { p1: "Amnesia-V2", p2: "FoulPlay-Minimax" },
    { p1: "Amnesia-V2", p2: "RandomPlayer" },
    { p1: "FoulPlay-Minimax", p2: "MaxDamage" }
  ];

  const startTime = Date.now();

  for (const m of matchups) {
    let p1Wins = 0;
    for (let i = 0; i < numGames; i++) {
      const res = await runGen8OUBattle(m.p1, m.p2);
      if (res === "p1") p1Wins++;
    }
    const wr = (p1Wins / numGames) * 100;
    console.log(`🎯 ${m.p1.padEnd(18)} vs ${m.p2.padEnd(18)}: ${wr.toFixed(1)}% Win Rate (${p1Wins}/${numGames})`);
  }

  const elapsed = (Date.now() - startTime) / 1000;
  console.log("=" .repeat(70));
  console.log(`⏱️ Completed ${matchups.length * numGames} Gen 8 OU battles in ${elapsed.toFixed(2)}s (${((matchups.length * numGames)/elapsed).toFixed(1)} battles/sec)`);
  console.log("=" .repeat(70));
}

if (require.main === module) {
  const games = parseInt(process.argv[2]) || 100;
  runGen8OULadderBenchmark(games);
}
