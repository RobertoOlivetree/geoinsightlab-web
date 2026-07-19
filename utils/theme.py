import html

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                max-width: 1450px;
                padding-top: 1.4rem;
                padding-bottom: 3rem;
            }

            [data-testid="stSidebar"] {
                border-right: 1px solid rgba(128, 128, 128, 0.22);
            }

            .geo-hero {
                padding: 2.6rem 2.7rem;
                margin-bottom: 1.4rem;
                border: 1px solid rgba(128, 128, 128, 0.22);
                border-radius: 22px;
                background:
                    radial-gradient(circle at top right, rgba(30, 144, 255, 0.18), transparent 34%),
                    linear-gradient(135deg, rgba(46, 139, 87, 0.18), rgba(14, 17, 23, 0.20));
            }

            .geo-hero h1 {
                font-size: clamp(2.4rem, 5vw, 4.7rem);
                margin: 0 0 0.35rem 0;
                line-height: 1.02;
            }

            .geo-hero h3 {
                margin: 0 0 1rem 0;
                font-weight: 500;
                opacity: 0.92;
            }

            .geo-hero p {
                max-width: 850px;
                font-size: 1.08rem;
                line-height: 1.65;
                margin-bottom: 0;
            }

            .geo-card {
                min-height: 170px;
                padding: 1.2rem 1.25rem;
                margin-bottom: 0.8rem;
                border: 1px solid rgba(128, 128, 128, 0.22);
                border-radius: 16px;
                background: rgba(255, 255, 255, 0.018);
            }

            .geo-card h3 {
                margin-top: 0;
            }

            .geo-note {
                padding: 0.95rem 1rem;
                margin: 0.7rem 0 1.2rem 0;
                border-left: 4px solid #2E8B57;
                border-radius: 7px;
                background: rgba(46, 139, 87, 0.10);
                line-height: 1.55;
            }

            .geo-workflow {
                padding: 1.1rem 1.2rem;
                border: 1px solid rgba(128, 128, 128, 0.22);
                border-radius: 14px;
                text-align: center;
                font-weight: 600;
                line-height: 1.75;
                letter-spacing: 0.01em;
            }

            .geo-status {
                display: inline-block;
                padding: 0.25rem 0.65rem;
                border-radius: 999px;
                background: rgba(46, 139, 87, 0.14);
                border: 1px solid rgba(46, 139, 87, 0.35);
                font-size: 0.86rem;
                font-weight: 600;
                margin-bottom: 0.9rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def scientific_note(text: str) -> None:
    st.markdown(
        f'<div class="geo-note">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def coming_soon(title: str, research_question: str, planned_features: list[str]) -> None:
    apply_theme()
    page_header(title, "A scientific module scheduled for a future GEOInsightLab release.")
    st.markdown('<span class="geo-status">In development</span>', unsafe_allow_html=True)
    st.subheader("Research question")
    scientific_note(research_question)
    st.subheader("Planned capabilities")
    for feature in planned_features:
        st.markdown(f"- {feature}")
    st.info(
        "This page is intentionally marked as in development. Results will only be "
        "published after the corresponding data, model artefacts and validation steps "
        "have been integrated and tested."
    )
