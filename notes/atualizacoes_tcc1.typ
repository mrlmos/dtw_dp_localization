#set text(lang: "pt", region: "br", size: 12pt)
#set par(justify: true, first-line-indent: (amount: 1.2em, all: true))


#align(center)[= Atualizações gerais]

#v(2em)

Nas ultimas 3 reuniões, eu comecei a aplicar os métodos desenvolvidos até o momento
em sinais sintéticos, para avaliar seu desempenho primeiro em situações mais controladas,
para eventualmente compará-los com dados reais. Na reunião da terça feira passada,
eu conversei com Tarso e George e concordamos em trazer os resultados nos sinais sintéticos
para o tcc1, e os reais para o tcc2.

Por conta disso, passei essas 2 semanas criando os sinais sintéticos e aplicando meus métodos
neles, pra escrever o texto. Por conta disso também estou apenas no começo da escrita, mas
ja consegui os resultados principais. Queria ajuda em como apresenta-los devidamente.

== Sinais sintéticos

O modelo de sinais sintéticos mais utilizado costumar ser o da exponencial dupla. Ele essencialmente
modela a descarga impulsiva que vemos em equipamentos elétricos, e adiciona um termo oscilatório.

Nos meus experimentos, eu controlei os valores de $beta$ enquanto mantinha o $alpha$
fixo, para replicar a distorção da forma de onda que acontece conforme a fonte se afasta
da antena. o quão mais próximo $beta$ se torna de $alpha$, mais o pulso ganha um formato
de gaussiana.
#figure(
  image("modelo_descarga.png"),
  caption: [Modelo analítico utilizado.],
)

#figure(
  image("ondas_geradas.png", width: 65%),
  caption: [Sinais gerados],
)<formas>

== Estrutura dos Testes

Eu determinei como objetivo analisar o desempenho dos métodos diante de circunstâncias
variadas (todas as defasagens foram mantidas fixas durante os testes):

+ *Formas de ondas iguais ($beta$ baixo) e SNR alto:* Todos os sinais tem o formato
  da primeira onda da @formas, e baixo nível de ruído.

+ *Formas de ondas iguais ($beta$ alto) e SNR alto:* Todos os sinais tem o formato
  da *última* onda da @formas, e baixo nível de ruído.
  O objetivo desses 2 testes é verificar se o método é invariante ao formato de onda,
  portanto que todas sejam iguais.

+ *Formas de ondas diferentes e SNR alto:* Essencialmente, os sinais da @formas, com
  formatos variados e baixo ruído. Testando o desempenho dos métodos quando existe
  distorção entre os sinais.

+ *Formas de ondas iguais ($beta$ baixo) e SNR variado:* Todos os sinais tem o formato
  da primeira onda da @formas, porém com níveis de ruído variados.

+ *Formas de ondas iguais ($beta$ alto) e SNR variado:* Todos os sinais tem o formato
  da *última* onda da @formas, com níveis de ruído variados.
  O objetivo desses 2 testes é verificar se o método é invariante ao formato de onda,
  portanto que todas sejam iguais.

+ *Formas de onda diferentes e SNR variado:* O teste final que simula o caso real de
  maneira mais próxima, com os níveis de ruido e deformação aumentando conforme a fonte
  se afasta da antena.

Os valores de SNR usados para o ruído branco aditivo foram de $15, 10, 5$ e $2$dB. Como referência, os métodos
clássicos de correlação cruzada desenvolvem suas formulações considerando um sinal
com baixo ruído, com SNR maior que 10dB.

Foram gerados 1000 (pois o DTW é muito lento) conjuntos de 4 sinais (defasagens fixas conforme a @formas), e
os métodos de Akaike, Correlação cruzada generalizada (GCC) e suas variações, Energia
cumulativa com tendência negativa (EnergyCrit, na minha nomeclatura), Energia cumulativa
com detecção de joelho por curvatura, e DTW foram aplicados e seus resultados foram
comparados.

== Resultados

De forma geral, o DTW foi consideravelmente melhor (ainda tenho que quantificar isso) do que os outros métodos nos
sinais sintéticos em quase todas as categorias, especialmente nas condições de alto
ruído e deformação, onde os outros métodos mostravam mais dificuldades.

=== Testes 1 e 2:

#figure(
  image("testes1e2.png"),
  caption: [Resultado dos testes 1 e 2. As barra pretas representam o intevalo de confiança de 95% das estimativas],
)

No primeiro caso, o dtw acertou perfeitamente todos os sinais, e no segundo
seu erro médio foi de $0.0061$. Enquanto Todos os métodos demonstram dificuldade
com os pulsos de formato gaussiano, o DTW mantem sua precisão.

No caso dos métodos baseados em energia, essa dificuldade ocorre pois, para
os sinais de formato gaussiano, o joelho da curva de energia cumulativa é mais
espalhado, dificultando a escolha consistente de um ponto de comparação.

Também é expressiva a alta variância dos métodos de correlação cruzada, algo
que se repete nos outros cenários.

=== Teste 3:

#figure(
  image("teste3.png", width: 70%),
  caption: [Resultado do teste 3.],
)

Para este caso, o DTW teve um erro médio levemente maior em comparação
ao GCC-PHAT e GCC-SCOT, porém apresentou uma variância significantemente menor.

=== Testes 4 e 5:

#figure(
  image("testes4e5.png"),
  caption: [Resultado dos testes 4 e 5.],
)

Aqui, o DTW se mostra significantemente melhor no segundo caso, reforçando
a ideia de que a técnica é menos sensível a sinais de densidade cumulativa
com joelhos "mal definidos".

=== Teste 6:

#figure(
  image("teste6.png", width: 70%),
  caption: [Resultado do teste 6.],
)

No caso mais Fiel a realidade, mais uma vez o DTW se mostra superior, tanto em
erro médio quanto em variância.

Uma outra observação importante é que nos casos onde o DTW errava, seu erro
médio absoluto apenas foi maior que $0.3$ amostras durante o alinhamento
dos últimos sinais, nos casos de alto ruído e deformação.
No ultimo teste, em especial, o erro médio de atraso de cada sinal foi de
$mat(delim: "[", 0.0, 0.0065, 4.2675)$ amostras, reforçando a ideia que a
imprecisão apenas se torna significativa nos casos extremos.
Por isso, para o caso de estimação de posição 2D, o ultimo sinal pode ser
ignorado (pois o sistema passa a ter uma equação a menos), o que explica
em partes a grande diferença de precisão de localização de 2D para 3D.

