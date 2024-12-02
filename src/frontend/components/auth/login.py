# src/frontend/components/auth/login.py
import streamlit as st
import bcrypt
from typing import Callable


class LoginComponent:
    def __init__(self, auth_callback: Callable[[str, str], bool]):
        self.auth_callback = auth_callback

    def render(self):
        st.markdown(
            """
        <style>
        .auth-form {
            max-width: 400px;
            margin: 0 auto;
            padding: 20px;
        }
        .stButton>button {
            width: 100%;
            margin-top: 10px;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        with st.container():
            st.markdown(
                "<h2 style='text-align: center;'>Login</h2>", unsafe_allow_html=True
            )

            with st.form("login_form", clear_on_submit=True):
                email = st.text_input("Email", key="login_email")
                password = st.text_input(
                    "Password", type="password", key="login_password"
                )

                col1, col2 = st.columns([1, 1])
                with col1:
                    submit = st.form_submit_button("Login")
                with col2:
                    st.form_submit_button(
                        "Register",
                        type="secondary",
                        on_click=lambda: st.session_state.update(
                            {"show_register": True, "show_login": False}
                        ),
                    )

                if submit and email and password:
                    if self.auth_callback(email, password):
                        st.session_state["authenticated"] = True
                        st.session_state["email"] = email
                        st.rerun()
                    else:
                        st.error("Invalid email or password")
