import streamlit as st
import bcrypt
from typing import Callable


class RegisterComponent:
    def __init__(self, register_callback: Callable[[str, str, str], bool]):
        self.register_callback = register_callback

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
                "<h2 style='text-align: center;'>Register</h2>", unsafe_allow_html=True
            )

            with st.form("register_form", clear_on_submit=True):
                name = st.text_input("Full Name", key="register_name")
                email = st.text_input("Email", key="register_email")
                password = st.text_input(
                    "Password", type="password", key="register_password"
                )
                confirm_password = st.text_input(
                    "Confirm Password", type="password", key="register_confirm_password"
                )

                col1, col2 = st.columns([1, 1])
                with col1:
                    submit = st.form_submit_button("Register")
                with col2:
                    st.form_submit_button(
                        "Back to Login",
                        type="secondary",
                        on_click=lambda: st.session_state.update(
                            {"show_register": False, "show_login": True}
                        ),
                    )

                if submit:
                    if not all([name, email, password, confirm_password]):
                        st.error("Please fill in all fields")
                    elif password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(password) < 8:
                        st.error("Password must be at least 8 characters long")
                    else:
                        if self.register_callback(name, email, password):
                            st.success("Registration successful! Please login.")
                            st.session_state.update(
                                {"show_register": False, "show_login": True}
                            )
                            st.rerun()
                        else:
                            st.error(
                                "Registration failed. Email might already be registered."
                            )
