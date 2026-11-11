#set document(title: "OMR Exam Grader — Visão geral do projeto", author: ("Rômulo", "Guilherme Oliver"))
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.2cm), numbering: "1 / 1")
#set text(lang: "pt", size: 10.5pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")
#show heading: it => block(above: 1.3em, below: 0.7em, it)
#show raw: set text(size: 9pt)

#let accent = rgb("#1f4e79")
#let box-step(body) = box(
  fill: accent.lighten(88%), stroke: accent, radius: 3pt,
  inset: (x: 7pt, y: 6pt), width: 100%, align(center, text(size: 8.5pt, body)),
)

#align(center)[
  #text(size: 20pt, weight: "bold", fill: accent)[OMR Exam Grader] \
  #v(2pt)
  #text(size: 12pt)[Correção automática de folhas de resposta por Optical Mark Recognition] \
  #v(6pt)
  #text(size: 10pt)[Processamento Digital de Imagens — UTFPR · IF69D / OPEET014 · 2026-2] \
  Rômulo e Guilherme Oliver
]

#v(6pt)
#line(length: 100%, stroke: 0.5pt + accent)

= O tema

*Optical Mark Recognition (OMR)* é a captura de dados a partir de marcas ópticas feitas em posições pré-determinadas de uma folha de papel. A aplicação escolhida é a correção de provas de múltipla escolha: o aluno marca a folha de respostas (FR), o professor fotografa com o celular e o sistema devolve a nota, sem leitora óptica dedicada.

*Objetivo.* Implementar um sistema capaz de: (1) adquirir a imagem da FR; (2) identificar as marcas e a resposta de cada questão; (3) comparar com um gabarito; (4) atribuir uma nota.

*Requisitos mínimos do enunciado:*
- FR própria com *50 questões × 5 alternativas* e até *4 tipos de prova*;
- procedimento para cadastrar o gabarito (um por tipo de prova) e o *valor de cada questão*;
- aquisição por *smartphone*; se usar app scanner, apenas correção de perspectiva (proibido converter para preto e branco);
- robustez a pequenas variações de posição, escala e ruído;
- código de terceiros, se houver, citado com destaque no relatório.

*Avaliação.* Na apresentação, o professor entrega FRs em branco para serem preenchidas e usadas para testar o sistema ao vivo. O sistema precisa funcionar com fotos novas, não só com o conjunto de desenvolvimento.

= Conteúdos da disciplina abordados

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + luma(170), inset: 5pt, align: (left, left),
  table.header([*Conteúdo (semana)*], [*Uso no projeto*]),
  [Imagem em níveis de cinza, binária e RGB (1)], [Conversão para cinza e binarização da foto da FR.],
  [Transformações geométricas e projetiva (2)], [Homografia que "planifica" a folha fotografada em perspectiva.],
  [Histograma e transformações ponto-a-ponto (3)], [Normalização de contraste/iluminação antes de binarizar.],
  [Filtros de vizinhança, Gaussiano e mediana (4, 5)], [Atenuação de ruído (sal-e-pimenta, granulação do papel).],
  [Detecção de bordas, Hough (6)], [Apoio à detecção de contornos da folha e dos marcadores de canto.],
  [Limiarização global (Otsu) e local, top-hat (9)], [Separar tinta e papel sob sombras e iluminação desigual.],
  [Morfologia, rotulação e componentes conexos (9)], [Localizar marcadores fiduciais e bolhas; filtrar por área e circularidade.],
  [Métricas objetivas e avaliação (5, 11)], [Acurácia de leitura, matriz de confusão e testes de robustez.],
)

= Visão geral do algoritmo

#block(breakable: false, grid(
  columns: (1fr, 12pt, 1fr, 12pt, 1fr),
  row-gutter: 8pt, align: horizon,
  box-step[*1. Aquisição* \ foto da FR com o celular],
  [→],
  box-step[*2. Pré-processamento* \ cinza, suavização, normalização],
  [→],
  box-step[*3. Binarização* \ Otsu e limiar adaptativo],
  box-step[*4. Marcadores de canto* \ morfologia + componentes conexos],
  [→],
  box-step[*5. Retificação* \ homografia para um gabarito de coordenadas],
  [→],
  box-step[*6. Amostragem da grade* \ ROI por bolha (50 × 5)],
  box-step[*7. Decisão de marcação* \ razão de preenchimento, branco/dupla],
  [→],
  box-step[*8. Tipo de prova* \ leitura das bolhas de tipo (A–D)],
  [→],
  box-step[*9. Correção* \ gabarito, pesos, nota final],
))

#v(4pt)
*Ideia central.* A FR é impressa com quatro marcadores quadrados nos cantos. Detectá-los na foto permite estimar a homografia $H$ que leva a folha fotografada ao sistema de coordenadas conhecido do modelo. Depois da retificação, cada bolha está numa posição previsível, e a decisão se reduz a medir *quanto da área da bolha está escura*. Duas ou mais bolhas acima do limiar na mesma questão contam como dupla marcação (anulada); nenhuma conta como em branco. O limiar de marcação é calibrado pela diferença entre bolhas preenchidas e vazias na própria folha, e não por um valor fixo, para tolerar variações de iluminação e de pressão da caneta.

= Stack

Linguagem *Python 3*, com as bibliotecas abaixo.

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + luma(170), inset: 5pt, align: (left, left),
  [*OpenCV (`opencv-python`)*], [Leitura de imagens, binarização, morfologia, componentes conexos, homografia e warp.],
  [*NumPy*], [Operações matriciais e cálculo de razão de preenchimento.],
  [*Matplotlib*], [Figuras das etapas intermediárias (relatório e slides) e gráficos de avaliação.],
  [*Pillow + ReportLab*], [Geração da FR em PDF (posições exatas, marcadores e bolhas).],
  [*Streamlit* _(opcional)_], [Interface simples para a demonstração: carregar foto, ver etapas, ver nota.],
  [*pytest*], [Testes automatizados de cada etapa com imagens sintéticas.],
)

Decisão de projeto: o núcleo (retificação, decisão de marcação) será implementado por nós com operações de baixo nível; o OpenCV entra para primitivas (`threshold`, `morphologyEx`, `findHomography`, `warpPerspective`). Qualquer trecho de terceiros será citado no relatório, como exige o enunciado.

= Etapas de desenvolvimento

O plano abaixo assume apresentação em *01/12* (entrega do relatório até 08/12); apresentar em 24/11 exigiria comprimir as fases 1–3 para cerca de uma semana.

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.4pt + luma(170), inset: 5pt, align: (left, left, left),
  table.header([*Fase*], [*Prazo*], [*Entregas*]),
  [*0. Preparação*], [11–12/11], [Repositório, ambiente Python, escolha e reserva do tema e do horário na planilha, divisão de tarefas.],
  [*1. Modelo da FR*], [12–15/11], [FR em PDF com 50 × 5 bolhas, 4 marcadores de canto, bloco de tipo de prova e identificação; arquivo com as coordenadas de cada bolha.],
  [*2. Banco de imagens*], [13–17/11], [Fotografar FRs preenchidas com variação de posição, escala, rotação, iluminação e sombra; registrar o gabarito verdadeiro (_ground truth_) de cada uma.],
  [*3. Retificação*], [15–19/11], [Binarização, detecção dos marcadores, homografia; validar com sobreposição da grade sobre a imagem retificada.],
  [*4. Leitura das marcas*], [18–22/11], [Amostragem por bolha, razão de preenchimento, tratamento de branco/dupla, leitura do tipo de prova.],
  [*5. Correção e interface*], [21–25/11], [Cadastro de gabaritos e pesos, cálculo da nota, saída em CSV; linha de comando e interface da demonstração.],
  [*6. Avaliação*], [24–28/11], [Acurácia por bolha e por folha, matriz de confusão e experimentos de robustez (rotação, escala, ruído, iluminação). Comparação Otsu × limiar adaptativo.],
  [*7. Relatório e slides*], [26/11–01/12], [Relatório em ~4 páginas no modelo da disciplina; slides com poucas palavras e muitas figuras, feitos por nós; ensaio da demonstração em até 18 min.],
  [*8. Entrega*], [até 08/12], [`.zip` público com código, imagens de teste, instruções, slides e relatório; link inserido no tópico "Entrega".],
)

= Avaliação planejada

- *Quantitativa:* acurácia por questão e por folha sobre o banco de imagens; matriz de confusão (marcada × lida); taxa de falha na detecção dos marcadores.
- *Robustez:* curvas de erro × ângulo de rotação, fator de escala, ruído gaussiano e sal-e-pimenta, e gradiente de iluminação.
- *Qualitativa:* figuras de cada etapa (foto, binária, marcadores, retificada, bolhas lidas sobrepostas, nota).
- *Teste ao vivo:* FRs preenchidas na hora pelo professor, fotografadas durante a apresentação.

= Riscos e mitigação

#table(
  columns: (1fr, 1fr),
  stroke: 0.4pt + luma(170), inset: 5pt,
  table.header([*Risco*], [*Mitigação*]),
  [Sombras e iluminação desigual na foto], [Limiar adaptativo e normalização de fundo; testar com muitas fotos ruins.],
  [Marcador de canto não detectado], [Filtros por área, razão de aspecto e posição; falhar com mensagem clara e pedir nova foto em vez de dar nota errada.],
  [Marcas fracas, rasuradas ou duplas], [Limiar relativo calibrado na própria folha; regra explícita para dupla marcação e para marca duvidosa.],
  [Foto muito inclinada ou com dedo cobrindo a folha], [Verificar a geometria da homografia (área e razão de aspecto) antes de ler as marcas.],
  [Falha na demonstração ao vivo], [Ensaiar com FRs novas, manter uma saída offline e conferir câmera e iluminação do local.],
)

= Estrutura do repositório

```
omr-exam-grader/
├── src/omr/          # pipeline: sheet, preprocess, markers, rectify, bubbles, grade
├── tools/            # geração da FR (PDF) e utilitários
├── data/             # fotos de teste e ground truth
├── tests/            # testes com imagens sintéticas
├── docs/             # este documento, relatório e slides
├── requirements.txt
└── README.md
```

#v(6pt)
#text(size: 9pt, fill: luma(100))[Referências iniciais: Wikipedia — Optical Mark Recognition; omrsolutions.com; Gonzalez & Woods, _Digital Image Processing_; documentação do OpenCV.]
