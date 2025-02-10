# Desenvolvendo um Robô de Batalha Pokemon

## Motivação

Desde sempre, eu fui um fã assíduo de pokemon. Em meados de 2014, conheci o pokemon Showdown, um simulador de batalhas pokemon online. O jogo é intuitivo e permite que os jogadores criem seus próprios times de pokemon e batalhem contra outros jogadores. Entretanto, ainda que não seja a parte mais atrativa da franquia pokemon, o competitivo e o lado estratégico dos jogos envolvidos é muito profundo e complexo.

Conforme fui me imergindo no competitivo, cada vez evolui mais e mais, num período de aprendizado que durou por volta de quatro anos. Infelizmente, com o tempo, fui me afastando do jogo. Hoje, estudando sobre algoritmos de Machine Learning, e a área de Inteligência Artificial, uma grande dúvida me veio à mente: Será que eu poderia criar um modelo que jogue de forma ótima o pokemon competitivo e gere resultados satisfatórios? É uma tarefa difícil, uma vez que o jogo é muito complexo e envolve muitas variáveis. Entretanto, acredito que seja um desafio interessante e que valha a pena ser explorado.

Ademais, acredito que o projeto possa ser uma ótima oportunidade para aprender mais sobre Machine Learning e Inteligência Artificial, além de ser uma forma de me reaproximar do jogo que tanto gosto.

## Objetivo

O objetivo do projeto é desenvolver um robô que jogue de forma ótima o pokemon competitivo. Para isso, será necessário criar um modelo de Machine Learning que seja capaz de aprender a jogar o jogo de forma eficiente. Para classificar os resultados do modelo, será utilizado a _ladder_, que é um sistema de ranking que classifica os jogadores de acordo com suas vitórias e derrotas. Dessa forma:
- Top 500: Excelente
- 1700: Satisfatório
- 1500: Regular
- Abaixo de 1500: Insatisfatório

Além disso, o projeto será dividido em duas etapas:
- O Cliente: Um sistema de _Web Scraping_ que será responsável por acessar o site do pokemon Showdown e coletar os dados necessários para o treinamento do modelo, além de futuramente fazer as ações do robô.
- O Modelo: Um modelo de Machine Learning que será treinado com os dados coletados pelo cliente e que será responsável por jogar o jogo de forma ótima.

O cliente será satisfatório se for capaz de coletar os dados necessários para o treinamento do modelo em larga escala e de forma eficiente. O modelo será satisfatório se for capaz de jogar o jogo de forma ótima e gerar resultados satisfatórios.

## Problemas iniciais


### Complexidade
O Pokémon competitivo pode ser comparado a diversos outros jogos de um contra um, como damas, alguns jogos de carta, ou xadrez. Entretanto, enquanto o xadrez possui 10^47 possibilidades de estados, o Pokémon possui 10^358 possibilidades de estados no turno inicial! Mantendo a analogia, uma vez que as peças de xadrez possuem diferentes comportamentos, o Pokémon possui mais de 800 peças diferentes, entretanto, cada um pode apresentar comportamentos individuais variados, como se fossem peças de xadrez que mudam de comportamento a cada partida. Isso exponencializa o número de possibilidades de estados do jogo.

É claro que o modelo não precisa estar _afiado_ para todas elas, mas isso se mostra um desafio para o projeto, uma vez que o modelo deve trabalhar com um espaço de estados muito grande. Veja, de forma resumida e simplificada, algumas das variáveis que o modelo deve considerar:
- Os climas(chuva, sol, etc)
- Os pokemons, e suas várias características(estatísticas, habilidades, tipos, etc)
- Os terrenos(que alteram o comportamento dos pokemons)
- Os _hazards_ (que causam dano passivo aos pokemons)
- O estado atual do jogo(vida dos pokemons, status, etc)
- Ações especiais(mega evolução, z-moves, etc)

### Predição
Assim como o xadrez, pokemon também envolve um alto grau de estratégia e planejamento. Isto é, o usuário deve prever os movimentos do adversário e se preparar e reagir a eles. Entretanto, diferente do xadrez, cada turno é constituído por ações dos dois jogadores simultaneamente, o que torna as predições mais dinâmicas e complexas. Por exemplo, se eu tiver um pokemon que é fraco contra um tipo de ataque, eu posso trocá-lo por outro pokemon que seja resistente a esse tipo de ataque. Entretanto, se o adversário prever essa troca e usar um ataque que seja super efetivo contra o pokemon que eu pretendo trocar, eu posso acabar perdendo o pokemon que eu pretendia salvar. Porém, eu também posso prever que o adversário vai prever a minha troca; E isso, na prática, gera um embate mental entre os jogadores.

O modelo deve ser capaz de aprender sobre predições e estar preparado para elas. Em alto nível, os jogadores tem um alto senso de predições, e isso é de alta importância para o sucesso no jogo. O modelo também não pode ser muito previsível, uma vez que o adversário pode se aproveitar disso.

### Tradução de dados
Muitos dados são numéricos, como as estatísticas dos pokemons, entretanto, muitos dados são categóricos, desde tipo dos pokemons, que são facilmente traduzidos para números, até habilidades e efeitos, que são mais complexos de serem traduzidos. O modelo deve ser capaz de lidar com esses dados de forma eficiente.

Por exemplo, a habilidade _Levitate_ faz com que o pokemon seja imune a ataques do tipo _Ground_. O modelo deve ser capaz de entender que um pokemon com a habilidade _Levitate_ não pode ser atingido por ataques do tipo _Ground_. Outro exemplo é o ataque _bullet seed_, que apesar de ter apenas 25 de poder, pode atingir até 5 vezes, o que resulta em um poder total de 125. O modelo deve ser capaz de entender que o ataque _bullet seed_, assim como outros ataques _multi-hit_, podem atingir mais de uma vez e não deve considerar apenas o poder base do ataque. São inúmeros exemplos de habilidades e efeitos que mudam completamente o comportamento do jogo e que o modelo deve ser capaz de lidar.

### Variabilidade dos formatos

Semelhante a campeonatos de artes marciais, o competitivo de pokemon possui vários formatos, que são regras diferentes que alteram o comportamento do jogo. Por exemplo, o formato _OU_ permite que os jogadores usem qualquer pokemon, enquanto o formato _Uber_ permite que os jogadores usem pokemons que são considerados muito fortes para o formato _OU_. Os citados precisam de um time pré-definido de 6 pokemons, enquanto outros formatos, como o _Random Battle_, dão um time aleatório de 6 pokemons para os jogadores. Deve-se escolher um formato para o modelo jogar.

Os modelos com times pré-definidos são mais fáceis de serem treinados, uma vez que o modelo pode aprender sobre os pokemons de forma menos abrangente. Entretanto, outros problemas surgem, como a escolha do time - que é um problema totalmente diferente de um modelo focado em batalhar - e menos demanda por batalhas. Os modelos com times aleatórios são mais difíceis de serem treinados, dado que trabalharão com diversos pokemons, mas terá mais demanda por batalhas.

Qual formato escolher? Como lidar com a variabilidade dos formatos? Como lidar com a variabilidade dos times?

### Compreensão de metagame

Assim como outros jogos, o Pokémon competitivo se luxa de um metagame diverso. Um metagame constituí de forma abstrata aqueles pokémons que, no determinado formato, se sobressaem sobre os outros em viabilidade, ou seja, o quão bom é ter aquele pokémon em seu time.

Mas, além disso, o pokémon como franquia traz a ideia de `gimmick`, ou seja, é de suma importância selecionar os pokémons que combinam entre si, sendo  um a cura da ferida do outro, por exemplo análogo.

Dessa forma, o modelo deve ter o "olho" que vai lhe indicar certas situações. Há pokémons que são altamente variáveis em suas usabilidades, como um Garchomo que pode ser usado como `sweeper` (nome dado aos pokémons que tem como função abater o maior número de pokémons), `speed control` (nome dado a aqueles que, por uso de itens ou habilidades, tem uma velocidade grande O ponto de ser uma arma ao mesmo tempo que uma válvula de escape, que pode ser mais rápido e evitar um _sweep_) Ou até mesmo um `tank` (nome dado aí pokémon que tem como função ser uma barreira defensiva). O modelo não pode agir da mesma forma para os três usos de Garchomo; Ou confundir e agir como se fosse um Garchomo tank quando ele é um sweeper, o que pode comprometer a partida.
### Adaptabilidade 
Outro papel importante do modelo é estar preparado para diferentes tipos de jogadores. Como citado na sessão `predição`, a mesma é uma estratégia que envolve risco. Logo, há jogadores que tomam mais riscos, os que jogam "agressivos" enquanto há aqueles que preferem tomar menos riscos e ter mais segurança. Uma vez que o modelo não deve ser altamente previsível, ele deve-se adaptar aos diferentes tipos de jogador.

Além dos tipos de jogador, há também aqueles mais habilidosos e os mais iniciantes, que jogam de forma completamente diferente, e portanto, seu oponente reage de uma forma diferente. É muito comum que, num contexto de combate entre dois iniciantes, estratégias simplórias consigam garantir vitórias nos primeiros dez turnos, como os `fears`, que é estratégia de utilizar um pokémon de nível um propositalmente pra enganar o adversário e garantir um abate.
## Metodologia

### Coleta de Dados

Apesar dos vários obstáculos citados acima, o nosso maior aliado é o fato de termos acesso a uma grande quantidade de dados. O site do pokemon Showdown possui uma API que nos permite acessar os dados de batalhas em alta quantidade. Além disso, temos dados relacionados a pokemons, habilidades, ataques, etc também, dentro do site: [https://play.pokemonshowdown.com/data/](https://play.pokemonshowdown.com/data/)

E com isso podemos trabalhar com pré-treinamento de modelos com batalhas antes existentes, além de acumular dados para treinamento de modelos futuros, se o algoritmo de treinamento for capaz de lidar com a quantidade de dados.

Apenas deve-se lidar com a formatação de dados, uma vez que alguns formatos são até mesmo em plain texts e devem ser formatados.

### Simulações iniciais

Ainda que o cliente seja uma interface de aplicação do modelo, a simulação de batalhas pode ser inicialmente delegada a uma biblioteca já existente, que o próprio pokemon Showdown disponibiliza. Viva o open source! A biblioteca `[pokemon-showdown](https://github.com/smogon/pokemon-showdown)` é uma biblioteca em Node.js que simula batalhas pokemon e que pode ser utilizada para testar o modelo. Entretanto, será um desafio fazer a comunicação entre o modelo e a biblioteca, uma vez que o modelo será desenvolvido em Python.

Entretanto, para uma performance melhor, deve-se tentar trabalhar com multithreading na simulação de batalhas e no aprendizado do modelo, uma vez que o modelo deve ser capaz de aprender em tempo real e de forma eficiente.

### Cliente

O cliente será desenvolvido em Python e utilizará algoritmos de Web Scraping para coletar os dados necessários para o treinamento do modelo; Ou seja, ele rodará um navegador e lerá as informações do site do pokemon Showdown, fazendo ações simples de um usuário como clicar num botão ou digitar num campo de texto. Entretanto, como a idealização de um web scrapping, tudo é automático por via de código. Para tal, será utilizado a biblioteca `selenium` para automatizar o navegador.

### Modelo

## Implementação da aplicação



## Resultados e Discussão

## Conclusão

## Referências

https://cs230.stanford.edu/projects_fall_2018/reports/12447633.pdf
http://julian.togelius.com/Lee2017Showdown.pdf
https://webthesis.biblio.polito.it/26843/

https://www.youtube.com/watch?v=rhvj7CmTRkg