"""Streamlit Interactive Live Demonstration UI for OMR Exam Grader (PDI — UTFPR).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))
if str(ROOT_DIR / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tools"))

from omr.analytics import compute_class_analytics
from omr.grade import AnswerKey, ExamConfig, create_default_exam_config
from omr.pipeline import OMRPipeline, PipelineConfig
from omr.sheet import EXAM_TYPES, OPTIONS, get_default_layout
from synthetic_dataset import SyntheticTransformParams, create_synthetic_sample


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_or_init_exam_config() -> ExamConfig:
    default_path = ROOT_DIR / "data" / "answer_keys" / "default_keys.json"
    if default_path.exists():
        return ExamConfig.load_json(default_path)
    return create_default_exam_config(seed=2026)


def main() -> None:
    st.set_page_config(
        page_title="OMR Exam Grader — PDI UTFPR",
        page_icon="📝",
        layout="wide",
    )

    st.title("📝 OMR Exam Grader — Correção Automática de Folhas de Respostas")
    st.caption(
        "**Processamento Digital de Imagens (IF69D / OPEET014) — UTFPR 2026.2** · "
        "Autores: Rômulo Barbosa da Silva e Guilherme Oliver"
    )

    if "exam_config" not in st.session_state:
        st.session_state["exam_config"] = _load_or_init_exam_config()

    exam_config: ExamConfig = st.session_state["exam_config"]

    # Sidebar: PDI Pipeline parameters
    with st.sidebar:
        st.header("⚙️ Parâmetros de PDI")
        binarize_method = st.selectbox(
            "Método de Binarização (Semana 9)",
            options=["adaptive", "otsu"],
            format_func=lambda x: "Limiar Adaptativo Gaussiano" if x == "adaptive" else "Limiar Global de Otsu",
        )
        denoise_method = st.selectbox(
            "Filtro de Vizinhança (Semanas 4–5)",
            options=["gaussian", "median", "bilateral", "none"],
            format_func=lambda x: {
                "gaussian": "Passa-Baixas Gaussiano (5×5)",
                "median": "Filtro da Mediana (5×5)",
                "bilateral": "Filtro Bilateral",
                "none": "Sem Filtragem",
            }[x],
        )
        normalize_illum = st.checkbox(
            "Normalização Morfológica de Fundo / Top-Hat",
            value=True,
            help="Divide a imagem pelo fundo estimado via fechamento morfológico para corrigir sombras do celular.",
        )
        adaptive_block = st.slider("Tamanho da Janela Adaptativa (px)", min_value=31, max_value=121, value=75, step=2)
        adaptive_c = st.slider("Constante C (Limiar Adaptativo)", min_value=5, max_value=35, value=18, step=1)
        fallback_type = st.selectbox("Tipo de Prova Padrão (caso em branco)", options=list(EXAM_TYPES), index=0)

        st.divider()
        pdf_path = ROOT_DIR / "data" / "templates" / "answer_sheet.pdf"
        if pdf_path.exists():
            st.download_button(
                label="📄 Baixar Folha de Respostas (PDF A4)",
                data=pdf_path.read_bytes(),
                file_name="folha_respostas_omr_utfpr.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    tab_corr, tab_batch, tab_keys = st.tabs(
        [
            "📸 Correção & Etapas do Pipeline",
            "📈 Correção em Lote & Análise Psicométrica (TCT)",
            "🔑 Cadastro de Gabaritos e Pesos",
        ]
    )

    with tab_keys:
        st.subheader("Configuração de Gabaritos (Tipos A, B, C, D) e Valor por Questão")
        col_k1, col_k2 = st.columns([2, 1])

        with col_k1:
            selected_type_edit = st.selectbox("Editar Gabarito do Tipo:", options=list(EXAM_TYPES))
            current_ans_str = "".join(
                exam_config.keys[selected_type_edit].answers[q] for q in range(1, 51)
            )
            new_ans_str = st.text_input(
                f"Sequência de 50 respostas (A–E) para a Prova Tipo {selected_type_edit}:",
                value=current_ans_str,
                max_chars=50,
            ).upper().strip()

            if st.button(f"Salvar Gabarito Tipo {selected_type_edit}"):
                if len(new_ans_str) == 50 and all(ch in OPTIONS for ch in new_ans_str):
                    exam_config.keys[selected_type_edit] = AnswerKey(
                        exam_type=selected_type_edit,
                        answers={q: new_ans_str[q - 1] for q in range(1, 51)},
                    )
                    st.success(f"Gabarito do Tipo {selected_type_edit} atualizado com sucesso!")
                else:
                    st.error("A sequência deve conter exatamente 50 letras entre A, B, C, D e E.")

        with col_k2:
            uniform_weight = st.number_input(
                "Peso uniforme por questão",
                min_value=0.05,
                max_value=10.0,
                value=float(exam_config.weights.get(1, 0.2)),
                step=0.05,
            )
            if st.button("Aplicar peso uniforme às 50 questões"):
                exam_config.weights = {q: float(uniform_weight) for q in range(1, 51)}
                st.success(f"Peso {uniform_weight:.2f} aplicado (Nota máxima = {exam_config.max_possible_score:.2f}).")

        st.json(exam_config.to_dict(), expanded=False)

    with tab_corr:
        source_mode = st.radio(
            "Fonte da Imagem da Folha de Respostas:",
            options=[
                "📤 Upload de Foto (Smartphone / Arquivo)",
                "📷 Câmera ao Vivo",
                "🧪 Simulador Interativo (Gerar Foto Sintética com Degradações)",
                "📂 Banco de Imagens de Teste (data/synthetic)",
            ],
            horizontal=True,
        )

        input_bgr: np.ndarray | None = None

        if source_mode.startswith("📤"):
            uploaded = st.file_uploader("Carregue uma foto da Folha de Respostas (.jpg, .png)", type=["jpg", "jpeg", "png", "bmp"])
            if uploaded is not None:
                file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                input_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        elif source_mode.startswith("📷"):
            cam_photo = st.camera_input("Fotografe a Folha de Respostas com os 4 marcadores visíveis")
            if cam_photo is not None:
                file_bytes = np.asarray(bytearray(cam_photo.read()), dtype=np.uint8)
                input_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        elif source_mode.startswith("🧪"):
            c1, c2, c3, c4, c5 = st.columns(5)
            sim_type = c1.selectbox("Tipo Marcado", options=list(EXAM_TYPES), index=0)
            sim_rot = c2.slider("Rotação 360° (°)", -180.0, 180.0, -5.0, 5.0)
            sim_shadow = c3.slider("Sombra (%)", 0, 70, 35, 5) / 100.0
            sim_sp = c4.slider("Sal e Pimenta (%)", 0.0, 6.0, 0.0, 0.5) / 100.0
            sim_seed = int(c5.number_input("Semente (Seed)", min_value=1, max_value=99999, value=2026))

            sample = create_synthetic_sample(
                "live_sim",
                exam_config=exam_config,
                exam_type=sim_type,
                params=SyntheticTransformParams(
                    rotation_deg=sim_rot,
                    scale_factor=0.86,
                    perspective_jitter_px=30.0,
                    shadow_strength=sim_shadow,
                    salt_pepper_prob=sim_sp,
                    gaussian_noise_sigma=4.0,
                ),
                seed=sim_seed,
            )
            input_bgr = sample.image_bgr

        else:
            synth_dir = ROOT_DIR / "data" / "synthetic"
            files = sorted(synth_dir.glob("*.png")) if synth_dir.exists() else []
            if files:
                chosen = st.selectbox("Selecione uma imagem de teste:", options=files, format_func=lambda p: p.name)
                raw = np.fromfile(str(chosen), dtype=np.uint8)
                input_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            else:
                st.warning("Nenhuma imagem encontrada em `data/synthetic/`. Execute `python tools/synthetic_dataset.py`.")

        if input_bgr is not None:
            cfg = PipelineConfig(
                denoise_method=denoise_method,
                normalize_illumination=normalize_illum,
                binarize_method=binarize_method,
                adaptive_block_size=adaptive_block,
                adaptive_c=adaptive_c,
                fallback_exam_type=fallback_type,
            )
            pipeline = OMRPipeline(config=cfg, layout=get_default_layout(), exam_config=exam_config)

            try:
                grading, art = pipeline.process_image(input_bgr)

                # Top Metrics Banner
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Tipo de Prova", grading.detected_exam_type or f"{grading.used_exam_type}*")
                m2.metric("Nota Final", f"{grading.total_score:.2f} / {grading.max_score:.2f}")
                m3.metric("Acertos (Verde)", f"{grading.num_correct} / 50")
                m4.metric("Erros (Vermelho)", f"{grading.num_wrong}")
                m5.metric("Em Branco", f"{grading.num_blank}")
                m6.metric("Dupla Marcação", f"{grading.num_multiple}")

                st.subheader("🔍 Visualização Passo a Passo das Etapas de PDI")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.image(_bgr_to_rgb(art.original_bgr), caption="1. Imagem Adquirida", use_container_width=True)
                    st.image(_bgr_to_rgb(art.markers.debug_image), caption="4. Detecção dos 4 Marcadores de Canto", use_container_width=True)
                with col_b:
                    st.image(art.preprocess.normalized, caption="2. Cinza + Suavização + Normalização de Fundo", use_container_width=True)
                    st.image(_bgr_to_rgb(art.rectification.grid_overlay), caption="5. Homografia (Retificada + Grade 50×5 ROIs)", use_container_width=True)
                with col_c:
                    st.image(art.preprocess.binary, caption=f"3. Binarização ({binarize_method.upper()})", use_container_width=True)
                    st.image(_bgr_to_rgb(art.graded_image), caption="6–9. Leitura das Bolhas e Correção Final", use_container_width=True)

                st.subheader("📊 Detalhamento por Questão")
                rows = []
                for q_num, qg in sorted(grading.questions.items()):
                    rows.append(
                        {
                            "Questão": f"{q_num:02d}",
                            "Gabarito": qg.expected_option or "-",
                            "Lida": ", ".join(qg.marked_options) if qg.marked_options else "(Branco)",
                            "Resultado": qg.outcome,
                            "Peso": f"{qg.weight:.2f}",
                            "Pontos": f"{qg.earned_points:.2f}",
                            "Fill A": f"{qg.fill_ratios.get('A', 0):.2f}",
                            "Fill B": f"{qg.fill_ratios.get('B', 0):.2f}",
                            "Fill C": f"{qg.fill_ratios.get('C', 0):.2f}",
                            "Fill D": f"{qg.fill_ratios.get('D', 0):.2f}",
                            "Fill E": f"{qg.fill_ratios.get('E', 0):.2f}",
                        }
                    )
                st.dataframe(rows, use_container_width=True, height=320)

                csv_buf = io.StringIO()
                writer = csv.DictWriter(csv_buf, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
                st.download_button(
                    "📥 Baixar Relatório da Folha (CSV)",
                    data=csv_buf.getvalue().encode("utf-8"),
                    file_name="relatorio_correcao_omr.csv",
                    mime="text/csv",
                )

            except Exception as exc:
                st.error(f"Erro durante o processamento da folha: {exc}")

    with tab_batch:
        st.subheader("📈 Correção de Turma em Lote & Análise Psicométrica (Teoria Clássica dos Testes)")
        st.markdown(
            "Avalia todas as folhas do banco de imagens (`data/synthetic`), calcula estatísticas da turma, "
            "**Índice de Dificuldade ($p_i$)**, **Índice de Discriminação ($D_i$)**, **Alfa de Cronbach ($\alpha$)** "
            "e sinaliza questões com possível erro de gabarito ou ambiguidade."
        )
        if st.button("🚀 Executar Correção em Lote e Análise de Itens (data/synthetic)", type="primary"):
            synth_dir = ROOT_DIR / "data" / "synthetic"
            files = sorted(synth_dir.glob("*.png")) if synth_dir.exists() else []
            if not files:
                st.warning("Nenhuma imagem em `data/synthetic/`.")
            else:
                cfg = PipelineConfig(
                    denoise_method=denoise_method,
                    normalize_illumination=normalize_illum,
                    binarize_method=binarize_method,
                    adaptive_block_size=adaptive_block,
                    adaptive_c=adaptive_c,
                    fallback_exam_type=fallback_type,
                )
                pipe = OMRPipeline(config=cfg, layout=get_default_layout(), exam_config=exam_config)
                results = []
                student_rows = []
                for fpath in files:
                    g, _ = pipe.process_file(fpath)
                    results.append(g)
                    student_rows.append(
                        {
                            "Arquivo": fpath.name,
                            "Tipo Lido": g.detected_exam_type or g.used_exam_type,
                            "Nota": f"{g.total_score:.2f} / {g.max_score:.2f}",
                            "Nota (0-10)": f"{g.score_normalized_10:.1f}",
                            "Acertos": g.num_correct,
                            "Erros": g.num_wrong,
                            "Brancos": g.num_blank,
                            "Duplas": g.num_multiple,
                        }
                    )

                report = compute_class_analytics(results)
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Alunos Avaliados", f"{report.num_students}")
                k2.metric("Média da Turma", f"{report.mean_score:.2f} (±{report.std_score:.2f})")
                k3.metric("Mediana", f"{report.median_score:.2f}")
                k4.metric("Taxa de Aprovação (≥6.0)", f"{report.pass_rate_pct:.1f}%")
                k5.metric("Alfa de Cronbach (α)", f"{report.cronbach_alpha:.3f}")

                st.markdown("#### Resultados por Folha")
                st.dataframe(student_rows, use_container_width=True)

                st.markdown("#### Análise Psicométrica por Questão (1 a 50)")
                item_rows = []
                for q_num, item in sorted(report.items.items()):
                    item_rows.append(
                        {
                            "Questão": f"{q_num:02d}",
                            "Gabarito": item.expected_option,
                            "Dificuldade (p_i)": f"{item.difficulty_index*100:.1f}%",
                            "Classificação": item.difficulty_category,
                            "Discriminação (D_i)": f"{item.discrimination_index:+.2f}",
                            "Mais Marcada": item.most_frequent_marked,
                            "Alerta Pedagógico": item.alert or "OK",
                        }
                    )
                st.dataframe(item_rows, use_container_width=True, height=320)


if __name__ == "__main__":
    main()
