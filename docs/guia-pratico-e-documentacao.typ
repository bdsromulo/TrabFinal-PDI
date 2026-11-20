#set document(
  title: "OMR Exam Grader — Documentação Completa do Código e Guia Prático de Uso",
  author: ("Rômulo Barbosa da Silva", "Guilherme Oliver"),
)
#set page(
  paper: "a4",
  margin: (x: 2.0cm, y: 2.0cm),
  numbering: "1 / 1",
)
#set text(lang: "pt", size: 10.2pt)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1.")
#show heading: it => block(above: 1.2em, below: 0.55em, it)
#show raw: set text(size: 8.8pt)

#let accent = rgb("#1f4e79")
#let callout(title, body) = block(
  fill: accent.lighten(90%),
  stroke: (left: 3pt + accent),
  inset: (x: 10pt, y: 8pt),
  radius: (right: 4pt),
  width: 100%,
  [*#title* \ #body],
)

#align(center)[
  #text(size: 19pt, weight: "bold", fill: accent)[OMR Exam Grader — Documentação Técnica e Guia Prático] \
  #v(3pt)
  #text(size: 11pt)[Manual de Arquitetura, Módulos Python e Roteiro de Testes para Rômulo e Guilherme Oliver] \
  #v(4pt)
  #text(size: 9.5pt, fill: luma(90))[Processamento Digital de Imagens (IF69D / OPEET014) — UTFPR · 2026.2]
]

#v(4pt)
#line(length: 100%, stroke: 0.6pt + accent)

= Guia Prático Rápido (Para Rômulo e Oliver Testarem Agora)

Este repositório já está 100% funcional e contém todo o material prático gerado (folha A4 em PDF para impressão, gabaritos de 4 tipos de prova, 12 fotos sintéticas de smartphone com degradações, interface visual em Streamlit e CLI).

== Comandos Principais do Dia a Dia

#table(
  columns: (1.2fr, 1.8fr),
  stroke: 0.4pt + luma(170),
  inset: 6pt,
  table.header([*O que você quer fazer*], [*Comando (dentro da pasta `omr-exam-grader/`)*]),
  [Abrir a interface visual (Demo ao vivo)], [`streamlit run app.py`],
  [Rodar todos os testes automatizados], [`python -m pytest -v`],
  [Corrigir uma foto via terminal (salvando as etapas)], [`python -m src.omr.cli data/synthetic/03_sombra_forte_tipoC.png --save-stages`],
  [Corrigir uma pasta inteira de fotos e gerar CSV], [`python -m src.omr.cli data/synthetic --output-dir output`],
  [Gerar novamente o PDF da Folha de Respostas], [`python tools/generate_sheet.py`],
  [Gerar o banco de 12 fotos sintéticas de teste], [`python tools/synthetic_dataset.py`],
  [Rodar os experimentos e gerar figuras do relatório], [`python tools/evaluate.py`],
)

== Como Imprimir e Testar com Fotos Reais do Celular de Vocês

1. *Imprimir a folha oficial:* Abra o arquivo `data/templates/answer_sheet.pdf` e imprima em folha A4 branca comum.
   - *Importante:* Na janela de impressão, selecione *"Tamanho real" (100%)* ou *"Ajustar à área imprimível"*. Mesmo que a impressora aplique uma leve escala, os *4 quadrados pretos de canto* ancoram a homografia automaticamente!
2. *Preencher a folha:*
   - Marque uma das bolhas de *Tipo de Prova* (`A`, `B`, `C` ou `D`) no canto superior direito;
   - Preencha as questões de `01` a `50` com caneta esferográfica preta ou azul escura (você pode deixar questões em branco e marcar duas bolhas na mesma questão para testar a anulação por dupla marcação).
3. *Fotografar com o celular:*
   - Apoie a folha sobre uma mesa plana;
   - Tire a foto garantindo que *os 4 quadrados pretos de canto apareçam inteiros* dentro do enquadramento;
   - *Regra do professor:* Não use filtro "preto e branco / documento" de apps de scanner (a foto deve vir colorida/cinza natural da câmera).
4. *Corrigir na hora:*
   - Rode `streamlit run app.py`, escolha *"Upload de Foto"* (ou *"Câmera ao Vivo"*), envie a foto e veja todas as 9 etapas de PDI e a nota na tela!

#callout("Roteiro para o Teste Ao Vivo com o Professor na Apresentação")[
  Durante a defesa, os professores Gustavo Borba / Humberto Gamba entregam folhas em branco ou pedem para preencher um gabarito na hora:
  1. Deixem o `streamlit run app.py` aberto no navegador.
  2. Na aba *"🔑 Cadastro de Gabaritos e Pesos"*, vocês podem digitar ou colar a sequência das 50 respostas do gabarito (`A`–`E`) para os Tipos A, B, C ou D e ajustar o peso das questões em segundos.
  3. Na aba *"📸 Correção & Etapas do Pipeline"*, carreguem a foto tirada pelo celular e mostrem os 6 painéis das etapas de PDI lado a lado + a tabela detalhada por questão.
]

= Arquitetura e Documentação Completa do Código (`src/omr/`)

O projeto foi desenhado com separação estrita de responsabilidades: cada etapa teórica de Processamento Digital de Imagens corresponde a um módulo Python independente em `src/omr/`.

== `src/omr/sheet.py` — Geometria Canônica da Folha (`SheetLayout`)

É a *fonte única de verdade* (*Single Source of Truth*) para as coordenadas físicas e digitais da Folha de Respostas:
- *Dimensões físicas:* Página A4 ($210" mm" times 297" mm"$), com origem $(0,0)$ no canto superior esquerdo.
- *Escala canônica:* `px_per_mm = 8.0` ($203,2" DPI"$), resultando em uma imagem canônica retificada de exatamente $1680 times 2376$ pixels. Como $1" mm" = 8" px"$, todas as coordenadas em milímetros convertem-se para pixels inteiros exatos sem erro de arredondamento.
- *Estruturas de dados:*
  - `MarkerCoord`: Define cada um dos 4 marcadores fiduciais quadrados (`size_mm = 14.0` mm, ou $112 times 112$ px), centrados em:
    - `top_left`: $(18" mm", 18" mm") arrow (144" px", 144" px")$
    - `top_right`: $(192" mm", 18" mm") arrow (1536" px", 144" px")$
    - `bottom_right`: $(192" mm", 279" mm") arrow (1536" px", 2232" px")$
    - `bottom_left`: $(18" mm", 279" mm") arrow (144" px", 2232" px")$
  - `BubbleCoord`: Define o centro $(c_x, c_y)$ e o raio $r$ de cada bolha:
    - *Bloco de Tipo de Prova (`exam_type_bubbles`):* 4 bolhas (`A`, `B`, `C`, `D`) em $y = 44" mm"$, $x in {138, 149, 160, 171}" mm"$, raio $r = 3,0" mm"$ ($24" px"$).
    - *Grade de 50 Questões (`question_bubbles`):* 250 bolhas divididas em 2 colunas de 25 questões (Questões 01–25 em $x_0 = 46" mm"$; Questões 26–50 em $x_0 = 132" mm"$), espaçamento horizontal entre alternativas $Delta x = 9,0" mm"$ ($72" px"$), espaçamento vertical entre questões $Delta y = 8,0" mm"$ ($64" px"$), e raio da bolha $r = 2,75" mm"$ ($22" px"$).
- *Função `render_sheet_image()`*: Renderiza em memória uma imagem BGR de $1680 times 2376$ px da folha em branco ou preenchida com marcações simuladas.

== `src/omr/preprocess.py` — Pré-processamento e Binarização (Etapas 2 e 3)

Implementa os conceitos das Semanas 1, 3, 4, 5 e 9 da disciplina:
1. *Conversão para níveis de cinza (`to_grayscale`):* Converte BGR para 1 canal `uint8` via `cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)`.
2. *Filtragem espacial de vizinhança (`apply_denoising`):*
   - `"gaussian"`: Filtro passa-baixas linear `cv2.GaussianBlur` ($5 times 5$) para suavizar ruído Gaussiano do sensor e granulação do papel.
   - `"median"`: Filtro não-linear `cv2.medianBlur` ($5 times 5$), ideal para eliminar ruído impulsivo (sal-e-pimenta) sem borrar as bordas dos marcadores.
3. *Normalização morfológica de iluminação (`normalize_illumination_background`):*
   - Estima o envelope de iluminação do papel branco aplicando fechamento morfológico (`cv2.MORPH_CLOSE`) com um elemento estruturante elíptico grande (~4,5% da menor dimensão da foto, tipicamente $75 times 75$ px), seguido de suavização Gaussiana.
   - Divide pixel a pixel a imagem original pelo fundo estimado (`cv2.divide(gray, background, scale=255)`), eliminando sombras diagonais ou vinhetas da câmera.
4. *Binarização (`binarize_image`):*
   - Inverte a polaridade (`THRESH_BINARY_INV`) para que *tinta/marcação = 255 (primeiro plano)* e *papel = 0 (fundo)*.
   - Suporta `"adaptive"` (`cv2.adaptiveThreshold` com janela Gaussiana $75 times 75$ e $C = 18$) e `"otsu"` (`cv2.THRESH_OTSU`).

== `src/omr/markers.py` — Detecção dos 4 Marcadores de Canto (Etapa 4)

Localiza os 4 quadrados fiduciais nos cantos usando Morfologia Matemática e Rotulação de Componentes Conexos (Semana 9):
1. *Limpeza morfológica e preenchimento de contornos:* Aplica fechamento $5 times 5$, preenche o interior de contornos externos quadrados que não tocam a borda da foto e executa abertura morfológica (`cv2.MORPH_OPEN`) para eliminar linhas finas, letras e círculos ocos das bolhas, preservando apenas regiões escuras maciças.
2. *Rotulação (`cv2.connectedComponentsWithStats`):* Extrai área, caixa delimitadora $(x, y, w, h)$ e centroide $(c_x, c_y)$ de cada componente conexo (conectividade-8).
3. *Filtragem geométrica de candidatos (`_extract_candidates`):*
   - Área relativa compatível com os quadrados fiduciais ($0,055%$ a $2,5%$ da área da foto, excluindo bolhas individuais);
   - Razão de aspecto próxima de um quadrado ($w / h in [0,48; 2,10]$);
   - Taxa de ocupação da caixa delimitadora (*extent*) $>= 0,48$ e *solidez convexa* ($A / A_"hull"$) $>= 0,80$;
   - Aproximação poligonal de Douglas-Peucker (`cv2.approxPolyDP`) próxima de 4 vértices.
4. *Seleção do quadrilátero ótimo (`_select_four_corners`):*
   - Associa os melhores candidatos aos 4 cantos da imagem na ordem canônica horária `[top_left, top_right, bottom_right, bottom_left]`;
   - Verifica se os 4 pontos formam um *quadrilátero estritamente convexo* (`_is_convex_quad` via produto vetorial das arestas), com área mínima de $12%$ da foto e consistência mútua entre as áreas dos 4 marcadores.
   - Caso a foto esteja cortada ou um marcador esteja obstruído por um dedo, lança `MarkerDetectionError` com mensagem explicativa em vez de produzir leitura silenciosamente errada.

== `src/omr/rectify.py` — Retificação Projetiva por Homografia (Etapa 5)

Aplica as transformações projetivas da Semana 2:
1. Estima a matriz de homografia $bold(H) in RR^(3 times 3)$ via `cv2.findHomography(src_pts, dst_pts)` mapeando os 4 centroides detectados na foto para os 4 centros canônicos `[(144, 144), (1536, 144), (1536, 2232), (144, 2232)]`.
2. Valida o condicionamento de $bold(H)$ (`validate_homography`: determinante não-nulo e magnitude projetiva dentro de limites físicos).
3. Executa `cv2.warpPerspective` com interpolação bilinear (`cv2.INTER_LINEAR`) para gerar a imagem retificada de $1680 times 2376$ px, normaliza o fundo e binariza a folha retificada, além de desenhar a grade de debug com as 254 ROIs circulares sobrepostas (`grid_overlay`).

== `src/omr/bubbles.py` — Amostragem de ROIs e Decisão Adaptativa (Etapas 6, 7 e 8)

Realiza a leitura das 250 bolhas de questões e das 4 bolhas de tipo de prova:
1. *Máscara circular interna (`sample_bubble_fill`):*
   - Como cada bolha tem um anel circular impresso na borda ($r = 22" px"$), a amostragem usa um disco interno de raio $r_"int" = 0,74 dot r approx 16" px"$ (~804 pixels de área interna). Assim, o contorno impresso da bolha fica 100% fora da máscara!
   - Calcula a razão de preenchimento $rho_(q,o) in [0, 1]$ (fração de pixels iguais a 255 dentro do disco interno).
2. *Calibração dinâmica do limiar na própria folha (`calibrate_mark_threshold`):*
   - Ordena os 254 valores de $rho$ da folha. Como pelo menos 75% das bolhas de uma folha são vazias, os 65% menores valores caracterizam a distribuição das bolhas vazias ($mu_"vazia", sigma_"vazia"$).
   - Define o piso estatístico $tau_"floor" = max(0.22, mu_"vazia" + 4.5 sigma_"vazia")$ (garantindo que uma folha 100% em branco tenha zero falsos positivos).
   - Quando há bolhas preenchidas ($rho >= tau_"floor"$), posiciona o limiar $tau_"folha"$ entre a média das bolhas vazias e a mediana das bolhas preenchidas:
     $ tau_"folha" = max(tau_"floor", mu_"vazia" + 0.42 (tilde(rho)_"cheia" - mu_"vazia")). $
3. *Regra de decisão por questão (`_decide_group`):*
   - `BLANK`: Nenhuma alternativa atingiu $tau_"folha"$;
   - `MARKED`: Exatamente 1 alternativa atingiu $tau_"folha"$ com margem de dominância clara;
   - `MULTIPLE`: 2 ou mais alternativas atingiram $tau_"folha"$ (ou uma segunda marcação está a menos de 12% da marcação principal), anulando a questão.

== `src/omr/grade.py` — Correção Ponderada e Diagnóstico Visual (Etapa 9)

- `ExamConfig` e `AnswerKey`: Armazenam os gabaritos dos 4 tipos de prova (`A`, `B`, `C`, `D`) e o dicionário de pesos individuais `weights[1..50]` (serializáveis em JSON).
- `grade_sheet()`: Identifica o tipo de prova lido, seleciona o gabarito correspondente, classifica cada questão em `CORRECT`, `WRONG`, `BLANK` ou `MULTIPLE` e soma os pontos obtidos.
- `render_graded_sheet()`: Desenha sobre a imagem retificada círculos coloridos de diagnóstico:
  - *Verde:* Marcação correta;
  - *Vermelho:* Marcação errada (com círculo azul indicando qual era a alternativa correta do gabarito);
  - *Laranja:* Dupla marcação (questão anulada);
  - *Amarelo-ouro:* Questão deixada em branco;
  - *Banner inferior:* Resumo com Tipo de Prova, Nota Final, Nota normalizada de 0 a 10 e contagens.

== `src/omr/pipeline.py`, `src/omr/cli.py`, `src/omr/analytics.py` e `app.py`

- `OMRPipeline`: Orquestra as etapas 1 a 9 em uma única chamada `pipeline.process_image(img)` ou `pipeline.process_file(path)`, devolvendo `(GradingResult, PipelineArtifacts)` com todas as imagens intermediárias preservadas.
- `src/omr/cli.py`: Interface de linha de comando para processar fotos individuais ou pastas inteiras, exportando `resultados.csv`, relatórios JSON por folha e (com `--save-stages`) todas as 9 imagens intermediárias.
- `src/omr/analytics.py`: Motor de Análise Psicométrica de Turma (Teoria Clássica dos Testes — TCT), calculando média, mediana, desvio-padrão, taxa de aprovação, *Índice de Dificuldade ($p_i$)*, *Índice de Discriminação ($D_i$)*, *Alfa de Cronbach ($alpha$)* e alertas automáticos de revisão de gabarito.
- `app.py`: Aplicação web interativa em Streamlit para a apresentação ao vivo.

= Extensões Avançadas e Casos de Uso de Estresse (`tests/test_omr_stress.py`)

Para elevar a robustez do sistema além dos requisitos básicos do enunciado, foram implementadas e testadas com 25 testes automatizados (`pytest`) as seguintes capacidades avançadas de PDI:

1. *Invariância a Rotação Completa de $360^degree$ ($0^degree, 45^degree, 90^degree, 180^degree, 270^degree, -135^degree$):*
   - Os candidatos a marcadores fiduciais são filtrados com métricas invariantes a rotação via caixa delimitadora mínima rotacionada (`cv2.minAreaRect`) e extração multinível (`cv2.RETR_LIST`), ordenados ciclicamente por ângulo polar $"atan2"(y - macron(y), x - macron(x))$.
   - Na retificação (`_disambiguate_cyclic_orientation`), o sistema avalia as 4 permutações cíclicas sobre um preview binarizado localmente (imune a sombras) comparando a densidade de tinta do cabeçalho superior com a margem inferior vazia, corrigindo fotos tiradas de cabeça para baixo ($180^degree$) ou na horizontal ($90^degree / 270^degree$).
2. *Conversão Cromática para Canetas Coloridas e Lápis sob Luz Quente (`to_grayscale(chromatic_boost=True)`):*
   - Combina a luminância ITU-R BT.601 com o canal mínimo $min(R, G, B)$. No papel branco acromático ($R approx G approx B$) o comportamento é idêntico ao nível de cinza padrão, mas para tintas coloridas (azul, vermelha, verde) o canal complementar escuro amplifica o contraste tinta-papel.
3. *Micro-Registro Local e Score Híbrido para Marcações Imperfeitas e Rasuras (`sample_bubble_fill` e `_decide_group`):*
   - Compensa pequenas curvaturas do papel ou distorção radial de lente (*barrel distortion*) buscando o máximo local de preenchimento dentro da margem radial segura da bolha.
   - Suporta marcações não-padronizadas feitas por alunos (como `X`, *checkmarks* `✓` e preenchimentos parciais ou a lápis fraco).
   - Distingue uma mancha leve de borracha (*erasure smudge*, quando há uma marca principal forte $>= 0,58$ e um resíduo fraco $<= 0,36$) de uma dupla marcação genuína (`MULTIPLE`).
4. *Layouts Parametrizáveis (`create_custom_layout` em `src/omr/sheet.py`):*
   - Permite gerar e corrigir folhas com número arbitrário de questões (de 5 a 100 questões), alternativas (2 a 5, ex.: V/F, A–D, A–E) e colunas (1 a 4 colunas, como simulados estilo vestibular de 80 ou 100 questões).
