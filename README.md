# OMR Exam Grader

Correção automática de folhas de respostas de múltipla escolha utilizando **Optical Mark Recognition (OMR)** e técnicas clássicas de **Processamento Digital de Imagens (PDI)**.  
Projeto final da disciplina **IF69D / OPEET014 — Processamento Digital de Imagens**, UTFPR, 2026-2.

Uma folha de respostas própria (**50 questões × 5 alternativas A–E**, com suporte a até **4 tipos de prova A–D** e **4 marcadores fiduciais quadrados nos cantos**) é fotografada com smartphone; o sistema pré-processa e normaliza a iluminação, binariza a imagem, detecta os marcadores por morfologia e componentes conexos, retifica a perspectiva por homografia, amostra as bolhas com limiar calibrado na própria folha e calcula a nota ponderada.

## Pipeline de 9 Etapas
1. **Aquisição:** Foto da folha de respostas tirada com smartphone.
2. **Pré-processamento:** Conversão para níveis de cinza, filtragem espacial (Gaussiano ou Mediana) e normalização morfológica de iluminação/fundo.
3. **Binarização:** Limiarização local adaptativa Gaussiana ou limiarização global de Otsu.
4. **Detecção dos Marcadores de Canto:** Operações morfológicas + rotulação de componentes conexos filtrados por área, razão de aspecto, solidez e geometria do quadrilátero convexo (`[TL, TR, BR, BL]`).
5. **Retificação Projetiva:** Estimativa da homografia $H \in \mathbb{R}^{3\times 3}$ e *warp* de perspectiva para o referencial canônico ($1680 \times 2376\text{ px}$, $8\text{ px/mm}$).
6. **Amostragem da Grade de Bolhas:** Máscara circular interna ($r_{\text{int}} = 0.74\,r$) nas $50 \times 5$ bolhas de questões e $4$ bolhas de tipo de prova.
7. **Decisão de Marcação Adaptativa:** Razão de preenchimento (*fill ratio*) com limiar calibrado na distribuição bimodal da própria folha (detectando marcação única, questão **em branco** ou **dupla marcação / anulada**).
8. **Leitura do Tipo de Prova:** Identificação automática do caderno (`A`, `B`, `C` ou `D`).
9. **Correção e Diagnóstico Visual:** Cruzamento com o gabarito e pesos individuais por questão, geração da imagem retificada anotada e exportação de relatório CSV/JSON.

## Stack
Python 3, OpenCV (`opencv-python`), NumPy, Matplotlib, Pillow, ReportLab, pytest e Streamlit (interface interativa para demonstração ao vivo).

## Instalação e Uso Rápido

```bash
pip install -r requirements.txt
```

### 1. Gerar a Folha de Respostas (PDF A4), Template Canônico e Gabaritos
```bash
python tools/generate_sheet.py
```
Gera `data/templates/answer_sheet.pdf` (pronto para impressão A4), `data/templates/reference_sheet.png`, `data/templates/sheet_layout.json` e `data/answer_keys/default_keys.json`.

### 2. Gerar Banco de Fotos Sintéticas com Degradações e Ground Truth
```bash
python tools/synthetic_dataset.py
```
Gera 12 imagens simulando fotos de smartphone (perspectiva, rotação, escala, sombra direcional, ruído Gaussiano e sal-e-pimenta) em `data/synthetic/`.

### 3. Executar a Correção via Linha de Comando (CLI)
```bash
# Corrigir uma única foto salvando todas as etapas intermediárias:
python -m src.omr.cli data/synthetic/02_perspectiva_tipoB.png --save-stages --output-dir output

# Corrigir um lote inteiro de fotos e exportar resultados.csv:
python -m src.omr.cli data/synthetic --output-dir output
```

### 4. Executar a Interface Interativa de Demonstração Ao Vivo (Streamlit)
```bash
streamlit run app.py
```

### 5. Executar os Testes Automatizados e a Avaliação Quantitativa
```bash
pytest
python tools/evaluate.py
```
O script `tools/evaluate.py` executa os experimentos quantitativos (Otsu vs. Limiar Adaptativo, matrizes de confusão em 7.500 bolhas e curvas de robustez) e salva as figuras de 300 DPI em `docs/relatorio/figuras/`.

## Estrutura do Repositório
- `src/omr/` — Módulos do pipeline (`sheet`, `preprocess`, `markers`, `rectify`, `bubbles`, `grade`, `pipeline`, `cli`, `analytics`)
- `tools/` — Gerador da FR em PDF (`generate_sheet.py`), simulador de fotos (`synthetic_dataset.py`) e suíte de avaliação (`evaluate.py`)
- `data/` — Templates PDF/PNG, gabaritos JSON e banco de imagens de teste
- `tests/` — Testes automatizados com `pytest`
- `docs/` — Visão geral (`omr-visao-geral.typ`/`.pdf`) e relatório técnico final no modelo LaTeX da disciplina (`docs/relatorio/`)
- `app.py` — Interface visual em Streamlit para a apresentação e teste ao vivo

Autores: **Rômulo Barbosa da Silva** e **Guilherme Oliver**.
