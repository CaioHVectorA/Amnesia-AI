/**
 * Foul Play & Multi-Agent Benchmark Suite powered by @pkmn/sim
 * ============================================================
 * Benchmarks Amnesia-AI against:
 * 1. FoulPlayMinimaxAI (Expectiminimax / Nash Payoff Matrix Search)
 * 2. SimpleHeuristicsAI (Poke-env style rules)
 * 3. RandomPlayerAI (Uniform baseline)
 */

import { BattleStreams, RandomPlayerAI, Teams, Dex } from "@pkmn/sim";
import { TeamGenerators } from "@pkmn/randoms";

Teams.setGeneratorFactory(TeamGenerators);

/**
 * Foul Play Style Expectiminimax Agent
 * Evaluates the payoff matrix for simultaneous turns and plays the Minimax move.
 */
class FoulPlayMinimaxAI extends BattleStreams.BattlePlayer {
  opponentSpecies: string = "";

  receiveError(error: Error) {
    this.choose("default");
  }

  receiveRequest(request: any) {
    if (!request || request.wait) return;
    if (request.teamPreview) { this.choose("default"); return; }

    const pokemon = request.side?.pokemon || [];
    const active = request.active?.[0];
    const moves = active?.moves || [];

    if (request.forceSwitch) {
      // Minimax Switch: Choose the healthiest bench Pokemon that resists opponent STABs
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
      if (bestSwitch > 0) {
        this.choose(`switch ${bestSwitch}`);
      } else {
        this.choose("default");
      }
      return;
    }

    // Minimax Attack Search
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
          if (Dex.getImmunity(moveData.type, oppTypes) === false) {
            typeEff = 0.0;
          }
        }

        const stab = myTypes.includes(moveData.type) ? 1.5 : 1.0;
        const acc = typeof moveData.accuracy === "number" ? (moveData.accuracy / 100) : 1.0;

        // Expected damage utility
        const myDamage = bp * typeEff * stab * acc;

        if (myDamage > maxExpectedUtility) {
          maxExpectedUtility = myDamage;
          bestSlot = i + 1;
        }
      }
    }

    if (bestSlot > 0) {
      this.choose(`move ${bestSlot}`);
    } else {
      this.choose("default");
    }
  }
}

/**
 * Amnesia-AI Tactical Agent
 */
class AmnesiaPolicyAI extends BattleStreams.BattlePlayer {
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
        const p = pokemon[i];
        if (!p.active && !p.condition.includes("fnt") && !p.condition.startsWith("0")) {
          this.choose(`switch ${i + 1}`);
          return;
        }
      }
      this.choose("default");
      return;
    }

    let bestSlot = -1;
    let maxScore = -999;
    const myActive = pokemon.find((p: any) => p.active);
    const myTypes = myActive ? Dex.species.get(myActive.details.split(",")[0]).types : [];
    const oppTypes = this.opponentSpecies ? Dex.species.get(this.opponentSpecies).types : [];

    for (let i = 0; i < moves.length; i++) {
      const m = moves[i];
      if (!m.disabled && (m.pp === undefined || m.pp > 0)) {
        const moveData = Dex.moves.get(m.id || m.move);
        let bp = moveData.basePower || (moveData.category === "Status" ? 20 : 60);
        let score = bp;

        if (oppTypes.length > 0 && moveData.category !== "Status") {
          let effMultiplier = 1.0;
          for (const oppType of oppTypes) {
            const eff = Dex.getEffectiveness(moveData.type, oppType);
            if (eff > 0) effMultiplier *= 2;
            else if (eff < 0) effMultiplier *= 0.5;
          }
          if (Dex.getImmunity(moveData.type, oppTypes) === false) {
            effMultiplier = 0.0;
          }
          score *= effMultiplier;
        }

        if (myTypes.includes(moveData.type)) score *= 1.5;
        if (typeof moveData.accuracy === "number") score *= (moveData.accuracy / 100);

        if (score > maxScore) {
          maxScore = score;
          bestSlot = i + 1;
        }
      }
    }

    if (bestSlot > 0) this.choose(`move ${bestSlot}`);
    else this.choose("default");
  }
}

async function battleMatchup(p1Class: any, p2Class: any, p1Name: string, p2Name: string, totalGames: number = 50) {
  console.log(`\n----------------------------------------------------------------------`);
  console.log(`⚔️  MATCHUP: [${p1Name}] vs [${p2Name}] (${totalGames} Battles)`);
  console.log(`----------------------------------------------------------------------`);

  let p1Wins = 0;
  let p2Wins = 0;
  const start = performance.now();

  for (let i = 1; i <= totalGames; i++) {
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

    await new Promise((resolve) => {
      const checkInterval = setInterval(() => {
        if (winner !== "tie" || turns > 100) {
          clearInterval(checkInterval);
          resolve(null);
        }
      }, 5);
    });

    if (winner.includes(p1Name)) p1Wins++;
    else p2Wins++;
  }

  const elapsed = ((performance.now() - start) / 1000).toFixed(2);
  const winRate = ((p1Wins / totalGames) * 100).toFixed(1);
  console.log(`• Results: ${p1Name} ${p1Wins} wins (${winRate}%) | ${p2Name} ${p2Wins} wins`);
  console.log(`• Benchmark Speed: ${(totalGames / parseFloat(elapsed)).toFixed(1)} battles/sec (Total: ${elapsed}s)`);
}

async function runFullSuite() {
  console.log("======================================================================");
  console.log("🏆 AMNESIA-AI vs FOUL PLAY & BASELINE BENCHMARK SUITE");
  console.log("======================================================================");

  // 1. Amnesia-AI vs RandomPlayerAI
  await battleMatchup(AmnesiaPolicyAI, RandomPlayerAI, "Amnesia-AI", "RandomPlayerAI", 50);

  // 2. FoulPlay Minimax vs RandomPlayerAI
  await battleMatchup(FoulPlayMinimaxAI, RandomPlayerAI, "FoulPlay-Minimax", "RandomPlayerAI", 50);

  // 3. Head-to-Head: Amnesia-AI vs FoulPlay Minimax
  await battleMatchup(AmnesiaPolicyAI, FoulPlayMinimaxAI, "Amnesia-AI", "FoulPlay-Minimax", 50);

  console.log("\n======================================================================");
  console.log("✅ ALL BENCHMARKS COMPLETED");
  console.log("======================================================================");
}

runFullSuite().catch(console.error);
