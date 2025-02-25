Um bom avanço, agora, uma forma de ler e traduzir o stream em texto para algo palpável desse ser feito.

Para entender e usar a lib @pkmn/sim, antes vou usar a lib original, mesmo que ainda falte typescript.

O que foi percebido:
- O simulador inteiro é em streams. As ações são escritas e os efeitos são recebidos.(Logo, tudo é texto.)
- Algumas coisas vem em formato JSON.
- Aparentemente, os formatos são estáveis.
- As ações são direcionadas entre p1 e p2.
- As teams e o metagame são definidos no início.

O próximo passo é trabalhar com normalizações, serializações. 

Foi separado entre `index.ts` utilizando a lib original e `sim.ts` utilizando a lib @pkmn/sim.

TODO:
- [ ] Formatos, tradução, tipagem
    - [ ] Reconhecer o formato
    - [ ] Normalizar as coisas, pensar nos dois lados
    - [ ] Traduzir o formato
    - [ ] Tipar o formato
- [ ] Dado uma forma mais organizada, pensar nos loggings e no env
- [ ] Ações