import requests
import csv
def seed(limit):
    url = f'https://pokeapi.co/api/v2/pokemon?limit={limit}'
    response = requests.get(url)
    data = response.json()
    poke_data = []
    for pokemon in data['results']:
        res = requests.get(pokemon['url']).json()
        index = int(pokemon['url'].split('/')[-2]) - 1
        # Abilities, name, id, stats.map(base_stat), types.map(type.name), egggroup
        # separate stat into columns and type1 type2
        formattedData = {}
        formattedData['name'] = res['name']
        formattedData['id'] = res['id']
        formattedData['abilities'] = [ability['ability']['name'] for ability in res['abilities']]
        formattedData['hp'] = res['stats'][0]['base_stat']
        formattedData['attack'] = res['stats'][1]['base_stat']
        formattedData['defense'] = res['stats'][2]['base_stat']
        formattedData['special-attack'] = res['stats'][3]['base_stat']
        formattedData['special-defense'] = res['stats'][4]['base_stat']
        formattedData['speed'] = res['stats'][5]['base_stat']
        formattedData['type1'] = res['types'][0]['type']['name']
        if len(res['types']) > 1:
            formattedData['type2'] = res['types'][1]['type']['name']
        else:
            formattedData['type2'] = None
        poke_data.append(formattedData)
    # write to csv file

    with open('pokemon.csv', mode='w') as file:
        fieldnames = ['name', 'id', 'abilities', 'hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed', 'type1', 'type2']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for pokemon in poke_data:
            writer.writerow(pokemon)