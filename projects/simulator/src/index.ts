import rl from "readline";
import Sim from "pokemon-showdown";
// const Sim = require("pokemon-showdown");
const stream = new Sim.BattleStream();
async function printNormalized(str: string) {
  if (str.includes("|request|")) {
    const req = str.split("|request|")[1];
    console.log("init", { str: str.split("|request|")[0] });
    await Bun.write(
      __dirname + "/moves/req.json",
      JSON.stringify(JSON.parse(req), null, 2)
    );
    // return console.log("req:", { str: JSON.parse(str.split("|request|")[1]) });
  }
  console.log(str);
}
(async () => {
  for await (const output of stream) {
    printNormalized(output);
    // console.log(output);
  }
})();

stream.write(`>start {"formatid":"gen7randombattle"}`);
stream.write(`>player p1 {"name":"Alice"}`);
stream.write(`>player p2 {"name":"Bob"}`);
// stream.write(`>p1 move 1`);
// stream.write(`>p2 move 1`);

let input = rl.createInterface({
  input: process.stdin,
  output: process.stdout,
  // terminal: false,
});
input.on("line", (line) => {
  stream.write(`>${line}`);
});
input.on("close", () => {
  stream.end();
});
