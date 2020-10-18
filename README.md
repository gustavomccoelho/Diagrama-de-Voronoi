# Diagrama de Voronoi

## Objetivo

O objetivo deste projeto é descrever as abordagens usadas na implementação de um Diagrama de Voronoi para um dado conjunto de pontos. Para a construção do diagrama, usaremos o método baseado no Algoritmo Incremental. 
Primeiro será feita uma breve descrição teórica do método e seus passos para a construção de cada célula. Em seguida, demonstraremos os resultados de sua implementação. 


## Método

A descrição teórica do método utilizado pode ser encontrada no relatório neste repositório.

## Implementação

A implementação do algoritmo descrito foi implementada em linguagem Python e executada em dois conjunto de pontos pré-definidos. Os resultados podem ser vistos nas figuras abaixo:

![nuvem_1](/pictures/nuvem_1.jpg)

![nuvem_2](/pictures/nuvem_2.jpg)

A variação do tempo de execução de acordo com o tamanho da entrada pode ser vista abaixo:

![runtime](/pictures/runtime.jpg)

## Conclusão

O algoritmo proposto foi implementado e como esperado, a complexidade assintótica do tempo de execução é  O_((n^2)). A implementação possui desafios principalmente em relação à erros de arredondamento no momento da definição da colinearidade entre uma aresta e os pontos de uma nova aresta adicionada. Para pontos que se encontram muito distantes da nuvem de pontos, o valor do determinante que define a colinearidade tende a ser maior, mesmo em pontos colineares. Em contrapartida, para pontos muito próximos mas não colineares, o determinante tende a ser pequeno. Estas discrepâncias tendem a induzir erros ao modelo, dependendo da distribuição dos pontos.

