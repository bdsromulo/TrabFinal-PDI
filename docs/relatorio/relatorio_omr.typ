// Modelo oficial IF69D / OPEET014 (UTFPR 2026.2) implementado em Typst
// Espaçamento simples, fonte Times 11pt, paginação n/m, cabeçalho com título resumido.

#set document(
  title: "OMR Exam Grader: Correção Automática de Folhas de Respostas por Reconhecimento Óptico de Marcas",
  author: ("Rômulo Barbosa da Silva", "Guilherme Oliver"),
)

#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.2cm, left: 2.5cm, right: 2.5cm),
  header: context {
    if counter(page).get().first() > 1 [
      #set text(font: ("Times New Roman", "Liberation Serif"), size: 9.5pt, style: "italic")
      OMR Exam Grader — Correção Automática de Folhas de Respostas por OMR
      #h(1fr)
      IF69D / OPEET014 — 2026.2
      #v(-4pt)
      #line(length: 100%, stroke: 0.4pt)
    ]
  },
  footer: context [
    #set text(font: ("Times New Roman", "Liberation Serif"), size: 10pt)
    #align(right)[#counter(page).display("1 / 1", both: true)]
  ],
)

#set text(
  font: ("Times New Roman", "Liberation Serif"),
  size: 11pt,
  lang: "pt",
)

#set par(
  justify: true,
  leading: 0.55em,
  first-line-indent: 1.0cm,
)

#set heading(numbering: "1.1.")
#show heading: it => {
  set par(first-line-indent: 0cm)
  block(above: 1.2em, below: 0.6em, text(weight: "bold", size: if it.level == 1 { 12pt } else { 11pt }, it))
}

#set math.equation(numbering: "(1)")
#show figure.where(kind: table): set figure.caption(position: top)

// Cabeçalho institucional conforme IF69D_modeloRelatorio_2026-2.docx
#block(width: 100%, inset: 0pt)[
  #set par(first-line-indent: 0cm, leading: 0.45em)
  #align(center)[
    *Universidade Tecnológica Federal do Paraná – UTFPR – Campus Curitiba* \
    Departamento Acadêmico de Eletrônica – DAELN \
    Bacharelado em Sistemas de Informação / Engenharia Eletrônica \
    Disciplina: IF69D e OPEET014 – Processamento Digital de Imagens — Semestre: 2026.2 \
    Profs.: Gustavo B. Borba e Humberto R. Gamba
    #v(8pt)
    #text(size: 13pt, weight: "bold")[RELATÓRIO TÉCNICO] \
    #v(4pt)
    #text(size: 13pt, weight: "bold")[OMR Exam Grader: Correção Automática de Folhas de Respostas por Reconhecimento Óptico de Marcas]
    #v(6pt)
    *Alunos:* Rômulo Barbosa da Silva e Guilherme Oliver \
    nov.2026
  ]
]

#v(4pt)
#line(length: 100%, stroke: 0.6pt)
#v(4pt)

= Objetivo

Implementar em Python um sistema completo de Reconhecimento Óptico de Marcas (_Optical Mark Recognition_ — OMR) para correção automática de folhas de respostas de múltipla escolha (50 questões $times$ 5 alternativas e até 4 tipos de prova) adquiridas por câmera de smartphone, contemplando geração da folha em PDF, pré-processamento, binarização, detecção de marcadores fiduciais, retificação projetiva por homografia, leitura adaptativa das bolhas e atribuição de nota ponderada.

= Fundamentação Teórica

O Reconhecimento Óptico de Marcas (OMR) consiste em detectar a presença ou ausência de marcações feitas a caneta ou lápis em posições pré-determinadas de um formulário impresso [1]. Quando a aquisição é realizada por câmera de smartphone sem suporte fixo, a imagem capturada sofre três classes principais de degradação: (i) distorção projetiva (perspectiva, rotação e escala); (ii) iluminação não-uniforme e sombras; e (iii) ruído do sensor óptico e granulação do papel.

== Pré-processamento, Filtragem e Normalização de Iluminação

Dada uma imagem colorida $I_"RGB"(x,y)$, converte-se inicialmente para níveis de cinza $I(x,y) in [0, 255]$. Para atenuar o ruído de alta frequência antes da segmentação, aplica-se um filtro passa-baixas Gaussiano ($G_sigma * I$) ou, na presença de ruído impulsivo do tipo sal-e-pimenta, o filtro não-linear da mediana em vizinhança $k times k$ [1].

Sob iluminação desigual causada pela sombra do smartphone, a intensidade observada pode ser modelada pelo produto entre a iluminância $L(x,y)$ (variação espacial lenta) e a refletância do papel/tinta $R(x,y)$ (variação rápida). Estima-se o fundo claro do papel $hat(L)(x,y)$ por meio do operador morfológico de fechamento com elemento estruturante elíptico $B$ de raio superior ao das bolhas e letras, conforme a equação (1) [1]:

$ I_"norm"(x,y) = min(255, 255 dot (I(x,y)) / ((I bullet B)(x,y))), quad "onde " (I bullet B) = (I plus.o B) minus.o B. $ <eq:bgnorm>

Na etapa de binarização, comparam-se duas abordagens clássicas: a limiarização global ótima de Otsu [2], que maximiza a variância interclasses $sigma_B^2(T)$ do histograma global, e a limiarização local adaptativa Gaussiana [4], na qual o limiar local $T(x,y)$ é dado pela média ponderada Gaussiana $mu_W(x,y)$ numa janela $W times W$ subtraída de uma constante $C$, conforme a equação (2):

$ B(x,y) = cases(
  255\, &"se " I_"norm"(x,y) < mu_W(x,y) - C,
  0\,   &"caso contrário."
) $ <eq:adaptive>

== Morfologia, Componentes Conexos e Retificação por Homografia

Para localizar a folha no espaço, imprimem-se quatro marcadores fiduciais quadrados sólidos ($14" mm" times 14" mm"$) nos cantos da área útil. Sobre a imagem binária $B(x,y)$, aplica-se abertura morfológica para eliminar traços finos e executa-se a rotulação de componentes conexos com conectividade-8 [1]. Cada componente candidato $k$ é filtrado por área $A_k$, razão de aspecto $w_k / h_k approx 1$ e solidez $S_k = A_k / A_"hull,k" >= 0,72$.

Obtidos os quatro centroides detectados $bold(p)_i = [x_i, y_i, 1]^T$ na foto e suas coordenadas canônicas conhecidas no modelo da folha $bold(p)'_i = [x'_i, y'_i, 1]^T$ ($i in {"TL", "TR", "BR", "BL"}$), estima-se a matriz de transformação projetiva (homografia) $bold(H) in RR^(3 times 3)$ com 8 graus de liberdade pela equação (3) [3]:

$ lambda_i vec(x'_i, y'_i, 1) = mat(h_11, h_12, h_13; h_21, h_22, h_23; h_31, h_32, h_33) vec(x_i, y_i, 1). $ <eq:homography>

A imagem retificada $I_"ret"(x',y')$ é reconstruída por mapeamento inverso (_inverse mapping_) com interpolação bilinear na resolução canônica de $1680 times 2376$ pixels ($8" px/mm"$ em folha A4).

== Amostragem das Bolhas e Decisão de Marcação Calibrada

No referencial canônico retificado, cada bolha da questão $q in {1, dots, 50}$ e alternativa $o in {"A", "B", "C", "D", "E"}$ possui centro $(c_x, c_y)$ e raio $r$ conhecidos. Para evitar que o anel impresso da própria bolha contamine a leitura, define-se uma máscara circular interna $M_(q,o)$ de raio $r_"int" = 0,74 r$ e calcula-se a razão de preenchimento (_fill ratio_) $rho_(q,o)$ conforme a equação (4):

$ rho_(q,o) = 1 / (|M_(q,o)|) sum_((x',y') in M_(q,o)) (B_"ret"(x',y')) / 255 in [0, 1]. $ <eq:fill>

Como pelo menos $75%$ das 250 bolhas de uma folha correspondem a alternativas não marcadas, os $65%$ menores valores de $rho_(q,o)$ na própria folha estimam a média $mu_"vazia"$ e o desvio padrão $sigma_"vazia"$ do fundo, calibrando o limiar $tau_"folha"$ entre o modo das bolhas vazias e a mediana das bolhas preenchidas (com piso mínimo $tau_"min" = 0,22$). Questões sem nenhuma alternativa com $rho_(q,o) >= tau_"folha"$ são classificadas como *em branco*; questões com duas ou mais alternativas acima do limiar são invalidadas por *dupla marcação*.

= Implementação

O sistema foi implementado em Python 3 de forma modular no repositório `omr-exam-grader`, utilizando as primitivas de baixo nível do OpenCV [5] e NumPy para as operações matriciais, além do ReportLab para geração vetorial da Folha de Respostas (FR) em PDF. A Figura 1 ilustra o resultado prático das etapas sequenciais do pipeline sobre uma imagem de teste com inclinação projetiva, rotação e gradiente de sombra.

#figure(
  image("figuras/fig_pipeline_etapas.png", width: 84%),
  caption: [Etapas do pipeline OMR implementado: (a) foto adquirida com distorção projetiva e sombra; (b) normalização morfológica de iluminação; (c) binarização adaptativa; (d) detecção dos 4 marcadores fiduciais; (e) retificação por homografia e grade canônica de ROIs ($50 times 5$); (f) leitura das marcas e correção anotada.],
)

Os arquivos entregues organizam-se nos seguintes módulos:
- *Núcleo do Pipeline (`src/omr/`)*: `sheet.py` (geometria canônica A4 a $8" px/mm"$); `preprocess.py` (filtros Gaussiano/mediana, divisão morfológica de fundo e binarização Otsu/Adaptativa); `markers.py` (morfologia, componentes conexos e validação do quadrilátero convexo); `rectify.py` (homografia via `cv2.findHomography` e `cv2.warpPerspective`); `bubbles.py` (amostragem com máscara circular interna e calibração automática de $tau_"folha"$); `grade.py` (gabaritos para tipos A–D, pesos por questão e diagnóstico visual); `pipeline.py` e `cli.py` (execução em linha de comando).
- *Ferramentas, Interface e Testes (`tools/`, `tests/`, `app.py`)*: `tools/generate_sheet.py` (gera o PDF `data/templates/answer_sheet.pdf` e gabaritos JSON); `tools/synthetic_dataset.py` (simulador fotométrico e projetivo com _ground truth_); `tools/evaluate.py` (experimentos quantitativos); `app.py` (interface interativa em Streamlit para demonstração ao vivo); e `tests/test_omr_pipeline.py` (testes automatizados com `pytest`).

= Resultados e conclusões

A avaliação quantitativa foi conduzida sobre um conjunto de benchmark de 30 folhas completas ($1.500$ questões e $7.500$ bolhas avaliadas) submetidas a variações combinadas de rotação ($plus.minus 15^degree$), escala ($0,72$ a $0,94$), distorção projetiva de perspectiva, gradiente direcional de sombra (até $50%$), variação na pressão da caneta e ruído aditivo. A Tabela 1 sintetiza o desempenho global obtido pelo pipeline.

#figure(
  kind: table,
  caption: [Desempenho quantitativo global do sistema OMR sobre o conjunto de avaliação ($N = 30$ folhas, $1.500$ questões, $7.500$ bolhas).],
  table(
    columns: (2fr, 1fr),
    stroke: 0.5pt,
    inset: 5pt,
    align: (left, center),
    table.header([*Métrica Avaliada*], [*Resultado Obtido*]),
    [Taxa de detecção dos 4 marcadores fiduciais], [$100,00%$ (30 / 30)],
    [Erro médio de localização dos cantos (RMSE)], [$0,07" pixels"$ ($approx 0,01" mm"$)],
    [Acurácia de identificação do Tipo de Prova (A–D)], [$100,00%$ (30 / 30)],
    [Acurácia de leitura por bolha individual ($N = 7.500$)], [$100,00%$ ($"F1-Score" = 100,00%$)],
    [Acurácia de decisão por questão ($N = 1.500$, incl. brancos e duplas)], [$100,00%$],
  ),
)

A Figura 2 apresenta o comparativo entre a limiarização global de Otsu aplicada diretamente sobre a imagem suavizada e a combinação de normalização morfológica de fundo com limiarização adaptativa local. Observa-se na Figura 2(b) que, sob sombreamento superior a $45%$, o limiar único de Otsu classifica a região sombreada da página como tinta (primeiro plano), degradando a acurácia para $63%$ (Figura 2(d)). Em contrapartida, a normalização de fundo aliada ao limiar adaptativo mantém $100%$ de acurácia mesmo quando a sombra reduz a iluminância local em mais de $70%$.

#figure(
  image("figuras/fig_otsu_vs_adaptativo.png", width: 94%),
  caption: [Comparação entre Otsu Global e Limiarização Adaptativa sob iluminação não-uniforme: (a) folha com forte gradiente de sombra ($62%$); (b) falha do Otsu global direto; (c) segmentação correta com normalização morfológica + limiar adaptativo; (d) curva de acurácia em função da intensidade da sombra.],
)

As matrizes de confusão por bolha e por questão (incluindo as classes _Branco_ e _Dupla marcação_) são apresentadas na Figura 3, confirmando a ausência de falsos positivos ou falsos negativos nas $7.500$ bolhas testadas. Na Figura 4, avaliam-se os limites operacionais do algoritmo sob degradações extremas: o sistema mantém $100%$ de acurácia para rotações de até $25^degree$, escalas entre $55%$ e $95%$ e ruído Gaussiano com $sigma <= 35$. No experimento de ruído impulsivo sal-e-pimenta (Figura 4(d)), comprova-se a superioridade teórica do filtro não-linear da mediana ($5 times 5$), que preserva $100%$ de acurácia até $8%$ de pixels corrompidos, enquanto o filtro linear Gaussiano sofre degradação severa em $8%$ por espalhar os impulsos para dentro das máscaras de amostragem.

#figure(
  image("figuras/fig_matriz_confusao.png", width: 84%),
  caption: [Matrizes de confusão sobre o lote de 30 folhas degradadas: (a) nível de bolha ($N = 7.500$); (b) nível de questão ($N = 1.500$, discriminando alternativas A–E, questões em branco e duplas marcações).],
)

#figure(
  image("figuras/fig_curvas_robustez.png", width: 84%),
  caption: [Curvas de robustez paramétrica: (a) ângulo de rotação; (b) fator de escala; (c) ruído Gaussiano ($sigma$); (d) ruído sal-e-pimenta comparando pré-filtragem Gaussiana vs. Mediana.],
)

Como limitações críticas e sugestões para continuidade do trabalho, destacam-se: (i) a homografia plana assume que a folha está apoiada sobre uma superfície aproximadamente plana; curvaturas cilíndricas acentuadas (folha segurada na mão no ar) podem deslocar as bolhas centrais em relação à grade projetiva, o que poderia ser mitigado adicionando marcadores fiduciais intermediários ao longo das margens laterais; e (ii) a inclusão de leitura óptica do RA do aluno via grade de dígitos OMR ou QR Code no cabeçalho permitiria associar automaticamente cada nota à pauta da turma.

#heading(numbering: none)[Referências]
#set par(first-line-indent: 0cm)

[1] Gonzalez, R. C., Woods, R. E., _Digital Image Processing_. 4th ed. Pearson, 2018.

[2] Otsu, N., "A Threshold Selection Method from Gray-Level Histograms." _IEEE Transactions on Systems, Man, and Cybernetics_, v. 9, n. 1, p. 62–66, 1979.

[3] Hartley, R., Zisserman, A., _Multiple View Geometry in Computer Vision_. 2nd ed. Cambridge University Press, 2004.

[4] Bradley, D., Roth, G., "Adaptive Thresholding Using the Integral Image." _Journal of Graphics Tools_, v. 12, n. 2, p. 13–21, 2007.

[5] Bradski, G., "The OpenCV Library." _Dr. Dobb's Journal of Software Tools_, v. 25, n. 11, p. 120–125, 2000.
