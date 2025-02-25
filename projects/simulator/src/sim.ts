import {
  Dex,
  BattleStreams,
  RandomPlayerAI,
  Teams,
  type PokemonSet,
} from "@pkmn/sim";
import { createInterface } from "readline";
import { TeamGenerators } from "@pkmn/randoms";
const rain = [
  "Pelipper||DampRock|Drizzle|Hurricane,Uturn,Defog,Roost|Sassy|248,,8,,252,|||||",
  "Seismitoad||LifeOrb|SwiftSwim|WeatherBall,EarthPower,IcyWind,FocusBlast|Modest|,,,252,4,252||,0,,,,|||",
  "Zapdos||HeavyDutyBoots|Static|Hurricane,Thunder,WeatherBall,Roost|Timid|,,,252,4,252||,0,,,,|||",
  "Ferrothorn||Leftovers|IronBarbs|StealthRock,LeechSeed,KnockOff,GyroBall|Relaxed|248,,148,,112,||,,,,,0|||",
  "Barraskewda||ChoiceBand|SwiftSwim|FlipTurn,Liquidation,Crunch,CloseCombat|Adamant|,252,,,4,252|||||",
  "Crawdaunt||LifeOrb|Adaptability|SwordsDance,AquaJet,Liquidation,KnockOff|Adamant|,252,,,4,252|||||",
].join("]");

const sun = [
  "Torkoal||HeatRock|Drought|StealthRock,LavaPlume,BodyPress,RapidSpin|Bold|248,,252,,8,|M|,,,,,0|||",
  "Venusaur||LifeOrb|Chlorophyll|Growth,GigaDrain,EarthPower,WeatherBall|Modest|,,,252,4,252|F|,0,,,,|||",
  "Heatran||ChoiceScarf|FlashFire|Eruption,MagmaStorm,EarthPower,FlashCannon|Timid|,,,252,4,252||,0,,,,|||",
  "Landorus-Therian||Leftovers|Intimidate|Earthquake,Toxic,Uturn,Defog|Careful|252,,,,240,16|M||||",
  "Peluche|Clefable|EjectButton|MagicGuard|Moonblast,KnockOff,SoftBoiled,HealingWish|Bold|252,,252,,4,|F||||",
  "Godpult|Dragapult|ChoiceSpecs|Infiltrator|ShadowBall,DracoMeteor,Uturn,Flamethrower|Modest|,,,252,4,252|M||||",
].join("]");
const p1spec = {
  name: "Sun",
  team: sun,
};
const p2spec = {
  name: "Rain",
  team: rain,
};

const streams = BattleStreams.getPlayerStreams(
  new BattleStreams.BattleStream()
);
const spec = { formatid: "gen8customgame" };

// const p1 = new RandomPlayerAI(streams.p1);
// const p2 = new RandomPlayerAI(streams.p2);
// void p1.start();
// void p2.start();

void (async () => {
  for await (const chunk of streams.omniscient) {
    console.log(chunk);
  }
})();

void streams.omniscient.write(`>start ${JSON.stringify(spec)}
  >player p1 ${JSON.stringify(p1spec)}
  >player p2 ${JSON.stringify(p2spec)}`);

const input = createInterface({
  input: process.stdin,
  output: process.stdout,
});
let turn = 0;
console.log("p1");
input.on("line", (line) => {
  turn++;
  if (turn % 2 === 0) {
    return streams.p2.write(`>${line}`);
  }
  streams.p1.write(`>${line}`);
  console.log("p2");
});
