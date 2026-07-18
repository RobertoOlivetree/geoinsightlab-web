import streamlit as st


def apply_theme() -> None:
    st.markdown(
        '''
        <style>
            .block-container {
                max-width: 1450px;
                padding-top: 1.5rem;
                padding-bottom: 3rem;
            }

            [data-testid="stSidebar"] {
                border-right: 1px solid rgba(128, 128, 128, 0.22);
            }

            .geo-hero {
                padding: 2rem 2.2rem;
                margin-bottom: 1.4rem;
                border: 1px solid rgba(128, 128, 128, 0.22);
                border-radius: 18px;
                background:
                    linear-gradient(
                        135deg,
                        rgba(46, 139, 87, 0.13),
                        rgba(30, 144, 255, 0.10)
                    );
            }

            .geo-card {
                min-height: 145px;
                padding: 1rem 1.1rem;
                margin-bottom: 0.8rem;
                border: 1px solid rgba(128, 128, 128, 0.22);
                border-radius: 14px;
            }

            .geo-note {
                padding: 0.9rem 1rem;
                margin: 0.7rem 0 1.2rem 0;
                border-left: 4px solid #2E8B57;
                border-radius: 6px;
                background: rgba(46, 139, 87, 0.09);
            }
        </style>
        ''',
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def scientific_note(text: str) -> None:
    st.markdown(
        f'<div class="geo-note">{text}</div>',
        unsafe_allow_html=True,
    )
