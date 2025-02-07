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

Ademais, o projeto será dividido em duas etapas:
- O Cliente: Um sistema de _Web Scraping_ que será responsável por acessar o site do pokemon Showdown e coletar os dados necessários para o treinamento do modelo, além de futuramente fazer as ações do robô.
- O Modelo: Um modelo de Machine Learning que será treinado com os dados coletados pelo cliente e que será responsável por jogar o jogo de forma ótima.

O cliente será satisfatório se for capaz de coletar os dados necessários para o treinamento do modelo em larga escala e de forma eficiente. O modelo será satisfatório se for capaz de jogar o jogo de forma ótima e gerar resultados satisfatórios.

## Metodologia

### Cliente

O cliente será desenvolvido em Python e utilizará algoritmos de Web Scraping para coletar os dados necessários para o treinamento do modelo; Ou seja, ele rodará um navegador e lerá as informações do site do pokemon Showdown, fazendo ações simples de um usuário como clicar num botão ou digitar num campo de texto. Entretanto, como a idealização de um web scrapping, tudo é automático por via de código. Para tal, será utilizado a biblioteca `selenium` para automatizar o navegador.

### Modelo

## Implementação da aplicação

## Resultados e Discussão

## Conclusão

## Referências