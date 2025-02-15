import json
import requests
import csv
def seed():
    url = f'https://play.pokemonshowdown.com/data/pokedex.json'
    response = requests.get(url)
    dict = response.json()
    # Convert to CSV file
    with open('pokedex.csv', mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['name', 'type1', 'type2', 'hp', 'atk', 'def', 'spa', 'spdef', 'spd', 'abilities', 'heightm', 'weightkg', 'color', 'canEvo', 'eggGroup1', 'eggGroup2'])
        for key, value in dict.items():
            # if (value.get('name').__contains__('Mega')): continue
            if (value.get('name').__contains__('Gmax')): continue
            types = value.get('types', [])
            if (types.__contains__('Bird')): continue
            type1 = types[0] if len(types) > 0 else ''
            type2 = types[1] if len(types) > 1 else ''
            
            baseStats = value.get('baseStats', {})
            hp = baseStats.get('hp', '')
            atk = baseStats.get('atk', '')
            defense = baseStats.get('def', '')
            spa = baseStats.get('spa', '')
            spdef = baseStats.get('spd', '')
            spd = baseStats.get('spe', '')
            
            eggGroups = value.get('eggGroups', [])
            eggGroup1 = eggGroups[0] if len(eggGroups) > 0 else ''
            eggGroup2 = eggGroups[1] if len(eggGroups) > 1 else ''
            
            canEvo = 1 if 'evos' in value else 0
            
            abilities = value.get('abilities', {})
            abilities_list = [abilities.get('0', ''), abilities.get('1', ''), abilities.get('H', '')]
            # rm empty strings
            abilities_list = list(filter(None, abilities_list))
            abilities_str = json.dumps(abilities_list)            
            writer.writerow([
                value.get('name', ''),
                type1,
                type2,
                hp,
                atk,
                defense,
                spa,
                spdef,
                spd,
                abilities_str,
                value.get('heightm', ''),
                value.get('weightkg', ''),
                value.get('color', ''),
                canEvo,
                eggGroup1,
                eggGroup2
            ])
    print('Seed data complete')
seed()