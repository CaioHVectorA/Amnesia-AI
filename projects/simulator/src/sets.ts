import { Sets } from "@pkmn/sets";
const formatTeam = (team: string) => {
  // format is showdown exportable
  // we want to recepit a team and return a team object
  // in general way, we can split by \n\n
  const separated = team.split("\n\n");
  return separated.map((poke) => Sets.pack(Sets.importSet(poke)));
};

const sunTeam = formatTeam(`Torkoal (M) @ Heat Rock  
Ability: Drought  
EVs: 248 HP / 252 Def / 8 SpD  
Bold Nature  
IVs: 0 Spe  
- Stealth Rock  
- Lava Plume  
- Body Press  
- Rapid Spin  

Venusaur (F) @ Life Orb  
Ability: Chlorophyll  
EVs: 252 SpA / 4 SpD / 252 Spe  
Modest Nature  
IVs: 0 Atk  
- Growth  
- Giga Drain  
- Earth Power  
- Weather Ball  

Heatran @ Choice Scarf  
Ability: Flash Fire  
EVs: 252 SpA / 4 SpD / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Eruption  
- Magma Storm  
- Earth Power  
- Flash Cannon  

Landorus-Therian (M) @ Leftovers  
Ability: Intimidate  
EVs: 252 HP / 240 SpD / 16 Spe  
Careful Nature  
- Earthquake  
- Toxic  
- U-turn  
- Defog  

Peluche (Clefable) (F) @ Eject Button  
Ability: Magic Guard  
EVs: 252 HP / 252 Def / 4 SpD  
Bold Nature  
- Moonblast  
- Knock Off  
- Soft-Boiled  
- Healing Wish  

Godpult (Dragapult) (M) @ Choice Specs  
Ability: Infiltrator  
EVs: 252 SpA / 4 SpD / 252 Spe  
Modest Nature  
- Shadow Ball  
- Draco Meteor  
- U-turn  
- Flamethrower  
`);

console.log(sunTeam);
Bun.write(
  "./src/sets/sun.json",
  JSON.stringify(
    {
      team: sunTeam,
      name: "Sun team",
    },
    null,
    2
  )
);
