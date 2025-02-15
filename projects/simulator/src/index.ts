import {
  Dex,
  BattleStreams,
  RandomPlayerAI,
  Teams,
  type PokemonSet,
} from "@pkmn/sim";
import { TeamGenerators } from "@pkmn/randoms";
import readline from "readline";

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

Teams.setGeneratorFactory(TeamGenerators);

const streams = BattleStreams.getPlayerStreams(
  new BattleStreams.BattleStream()
);
const spec = { formatid: "gen8customgame" };
// console.log(Teams.pack(Teams.generate("gen8randombattle")));
const p1spec = {
  name: "Sun",
  team: sun,
};
const p2spec = {
  name: "Rain",
  team: rain,
};

// Mostra no terminal todas as mensagens enviadas pelo simulador
(async () => {
  for await (const chunk of streams.omniscient) {
    console.log(chunk);
  }
})();

// Inicializa a batalha com comandos pré-definidos
streams.omniscient.write(`>start ${JSON.stringify(spec)}
>player p1 ${JSON.stringify(p1spec)}
>player p2 ${JSON.stringify(p2spec)}\n`);

// Configuração do readline para ler comandos do usuário via CLI
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  prompt: ">", // Prompt para indicar onde digitar
});

rl.prompt();
let count = 0;
rl.on("line", (line) => {
  // Envia o comando digitado para o stream do simulador
  if (count % 2 === 0) {
    streams.p1.write(line.trim() + "\n");
  } else {
    streams.p2.write(line.trim() + "\n");
  }
  rl.prompt();
}).on("close", () => {
  console.log("Encerrando a interface CLI.");
  process.exit(0);
});
