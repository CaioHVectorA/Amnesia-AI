/**
 * Headless Simulation Benchmark powered by @pkmn/sim
 * ===================================================
 * Runs official headless Gen 9 Random Battles between:
 * - P1: Smart Policy Agent (Type-Effectiveness & STAB Maximizer)
 * - P2: RandomPlayerAI (Official Showdown baseline)
 */

import { BattleStreams, RandomPlayerAI, Teams, Dex } from "@pkmn/sim";
import { TeamGenerators } from "@pkmn/randoms";

Teams.setGeneratorFactory(TeamGenerators);

class SmartPolicyPlayerAI extends BattleStreams.BattlePlayer {
  opponentSpecies: string = "";

  receiveError(error: Error) {
    this.choose("default");
  }

  receiveRequest(request: any) {
    if (!request || request.wait) return;

    if (request.teamPreview) {
      this.choose("default");
      return;
    }

    const pokemon = request.side?.pokemon || [];
    const active = request.active?.[0];
    const moves = active?.moves || [];

    // 1. Forced Switch
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

    // 2. Tactical Move Selection (Type Effectiveness + STAB + BP)
    let bestSlot = -1;
    let maxScore = -100;

    const myActive = pokemon.find((p: any) => p.active);
    const myTypes = myActive ? Dex.species.get(myActive.details.split(",")[0]).types : [];
    const oppTypes = this.opponentSpecies ? Dex.species.get(this.opponentSpecies).types : [];

    for (let i = 0; i < moves.length; i++) {
      const m = moves[i];
      if (!m.disabled && (m.pp === undefined || m.pp > 0)) {
        const moveData = Dex.moves.get(m.id || m.move);
        let bp = moveData.basePower || (moveData.category === "Status" ? 20 : 60);
        let score = bp;

        // Type Effectiveness against Opponent
        if (oppTypes.length > 0 && moveData.category !== "Status") {
          let effMultiplier = 1.0;
          for (const oppType of oppTypes) {
            const eff = Dex.getEffectiveness(moveData.type, oppType);
            if (eff > 0) effMultiplier *= 2;
            else if (eff < 0) effMultiplier *= 0.5;
          }
          // Immunity check
          if (Dex.getImmunity(moveData.type, oppTypes) === false) {
            effMultiplier = 0.0;
          }
          score *= effMultiplier;
        }

        // STAB Bonus
        if (myTypes.includes(moveData.type)) {
          score *= 1.5;
        }

        // Accuracy weight
        if (typeof moveData.accuracy === "number") {
          score *= (moveData.accuracy / 100);
        }

        if (score > maxScore) {
          maxScore = score;
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

async function runSingleBattle(): Promise<{ winner: string; turns: number }> {
  const streams = BattleStreams.getPlayerStreams(new BattleStreams.BattleStream());
  const spec = { formatid: "gen9randombattle" };
  const p1spec = { name: "Amnesia-Smart" };
  const p2spec = { name: "RandomPlayerAI" };

  const p1 = new SmartPolicyPlayerAI(streams.p1);
  const p2 = new RandomPlayerAI(streams.p2);

  void p1.start();
  void p2.start();

  let winner = "tie";
  let turns = 0;

  void (async () => {
    for await (const chunk of streams.omniscient) {
      const lines = chunk.split("\n");
      for (const line of lines) {
        if (line.startsWith("|switch|p2a:") || line.startsWith("|drag|p2a:")) {
          p1.opponentSpecies = line.split("|")[3].split(",")[0].trim();
        } else if (line.startsWith("|turn|")) {
          turns = parseInt(line.split("|")[2], 10);
        } else if (line.startsWith("|win|")) {
          winner = line.split("|")[2].trim();
        }
      }
    }
  })();

  await streams.omniscient.write(`>start ${JSON.stringify(spec)}
>player p1 ${JSON.stringify(p1spec)}
>player p2 ${JSON.stringify(p2spec)}`);

  return new Promise((resolve) => {
    const checkInterval = setInterval(() => {
      if (winner !== "tie" || turns > 100) {
        clearInterval(checkInterval);
        resolve({ winner, turns });
      }
    }, 5);
  });
}

async function main() {
  const TOTAL_GAMES = 100;
  console.log("======================================================================");
  console.log("⚡ OFFICIAL HEADLESS @pkmn/sim BENCHMARK (Gen 9 Random Battles)");
  console.log("======================================================================");
  console.log(`Simulating ${TOTAL_GAMES} full official matches against RandomPlayerAI...\n`);

  let p1Wins = 0;
  let p2Wins = 0;
  const turnsList: number[] = [];
  const start = performance.now();

  for (let i = 1; i <= TOTAL_GAMES; i++) {
    const { winner, turns } = await runSingleBattle();
    if (winner.includes("Amnesia")) {
      p1Wins++;
    } else {
      p2Wins++;
    }
    turnsList.push(turns);

    if (i % 20 === 0 || i === TOTAL_GAMES) {
      const wr = ((p1Wins / i) * 100).toFixed(1);
      const avgTurns = (turnsList.reduce((a, b) => a + b, 0) / turnsList.length).toFixed(1);
      console.log(`[${i.toString().padStart(3, "0")}/${TOTAL_GAMES}] Amnesia-AI Wins: ${p1Wins.toString().padStart(3, " ")} | Win Rate: ${wr}% | Avg Turns: ${avgTurns}`);
    }
  }

  const elapsed = ((performance.now() - start) / 1000).toFixed(2);
  const finalWR = ((p1Wins / TOTAL_GAMES) * 100).toFixed(2);

  console.log("\n======================================================================");
  console.log("🏆 OFFICIAL BENCHMARK SUMMARY");
  console.log("======================================================================");
  console.log(`Total Battles Played:   ${TOTAL_GAMES}`);
  console.log(`Amnesia-AI Wins:        ${p1Wins} / ${TOTAL_GAMES}`);
  console.log(`RandomPlayerAI Wins:    ${p2Wins} / ${TOTAL_GAMES}`);
  console.log(`Win Rate vs Baseline:   ${finalWR}%`);
  console.log(`Average Turns/Battle:   ${(turnsList.reduce((a, b) => a + b, 0) / turnsList.length).toFixed(1)}`);
  console.log(`Total Simulation Time:  ${elapsed}s (${(TOTAL_GAMES / parseFloat(elapsed)).toFixed(1)} battles/sec)`);
  console.log("======================================================================");
}

main().catch(console.error);
