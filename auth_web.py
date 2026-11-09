from html import escape
from typing import Any

import gradio as gr
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from auth_store import AuthStore, DuplicateAccountError, RegistrationError, SESSION_COOKIE


_AUTH_CSS = """
<style>
    :root {
        color-scheme: light;
        --cream: #f4f1e5;
        --paper: #faf8ef;
        --forest: #18251d;
        --olive: #777451;
        --border: #d8d5c5;
        --muted: #65685e;
    }
    * { box-sizing: border-box; }
    body {
        margin: 0;
        min-height: 100vh;
        padding: 28px 20px;
        background: var(--cream);
        color: var(--forest);
        font-family: "DM Sans", "Segoe UI", sans-serif;
    }
    .page { width: min(100%, 480px); margin: 0 auto; }
    .brand {
        margin: 14px 0 34px;
        color: var(--forest);
        font-family: Georgia, "Times New Roman", serif;
        font-size: 30px;
        font-weight: 600;
        letter-spacing: -1px;
        text-align: center;
        text-decoration: none;
    }
    .brand span { color: var(--olive); font-style: italic; }
    .card {
        padding: clamp(24px, 6vw, 42px);
        border: 1px solid var(--border);
        border-radius: 22px;
        background: var(--paper);
        box-shadow: 0 12px 36px rgb(24 37 29 / 6%);
    }
    h1 {
        margin: 0 0 10px;
        font-family: Georgia, "Times New Roman", serif;
        font-size: clamp(30px, 7vw, 38px);
        font-weight: 500;
        letter-spacing: -1px;
    }
    .intro, .privacy, .switch { color: var(--muted); line-height: 1.65; }
    .intro { margin: 0 0 26px; }
    label { display: block; margin: 17px 0 7px; font-size: 14px; font-weight: 600; }
    input {
        width: 100%;
        min-height: 48px;
        padding: 11px 86px 11px 13px;
        border: 1px solid #aaa995;
        border-radius: 10px;
        background: #fffef9;
        color: var(--forest);
        font: inherit;
    }
    .password-field { position: relative; }
    .password-toggle {
        position: absolute;
        top: 50%;
        right: 9px;
        width: auto;
        min-height: 0;
        margin: 0;
        padding: 7px 9px;
        transform: translateY(-50%);
        border-radius: 7px;
        background: transparent;
        color: var(--forest);
        font-size: 13px;
    }
    .password-toggle:hover { background: #efeddf; color: var(--forest); }
    .password-hint { margin: 8px 0 0; color: var(--muted); font-size: 13px; line-height: 1.5; }
    .password-checklist {
        display: grid;
        gap: 7px;
        margin: 12px 0 0;
        padding: 13px 15px;
        border: 1px solid var(--border);
        border-radius: 10px;
        background: #fffef9;
        color: var(--muted);
        font-size: 13px;
        list-style: none;
    }
    .password-checklist li { display: flex; gap: 9px; align-items: baseline; }
    .password-checklist li.is-met { color: #355b3c; }
    .requirement-indicator { width: 14px; font-weight: 700; }
    .password-status { margin: 8px 0 0; color: var(--muted); font-size: 12px; }
    input:focus-visible, button:focus-visible, a:focus-visible {
        outline: 3px solid #777451;
        outline-offset: 3px;
    }
    button {
        width: 100%;
        min-height: 50px;
        margin-top: 24px;
        border: 0;
        border-radius: 999px;
        background: var(--forest);
        color: white;
        cursor: pointer;
        font: inherit;
        font-weight: 650;
    }
    button:hover { background: #304737; }
    .message {
        margin: 18px 0 0;
        padding: 12px 14px;
        border: 1px solid #c8a9a0;
        border-radius: 10px;
        background: #fff4ef;
        color: #713e31;
        line-height: 1.5;
    }
    .switch { margin: 22px 0 0; text-align: center; }
    a { color: #394b3d; font-weight: 600; }
    .privacy { margin: 22px 0 0; font-size: 13px; text-align: center; }
    @media (max-width: 480px) { body { padding: 16px 14px; } .brand { margin: 12px 0 24px; } }
</style>
"""

_AUTH_CSS = _AUTH_CSS.replace(
    "</style>",
    """
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    :root {
        --cream: #101116;
        --paper: #ffffff;
        --forest: #15161b;
        --olive: #4158f5;
        --border: #e1e3e9;
        --muted: #656a76;
    }
    body {
        background: #101116;
        color: var(--forest);
        font-family: "DM Sans", "Segoe UI", sans-serif;
    }
    .page { max-width: 460px; padding-top: 12px; }
    .brand { display: block; color: #fff; font-family: "DM Sans", sans-serif; }
    .brand span { color: #9eabff; font-style: normal; }
    .card {
        border-color: #e6e8ed;
        border-radius: 18px;
        background: #fff;
        box-shadow: 0 22px 60px rgb(0 0 0 / 24%);
    }
    h1 { font-family: "DM Sans", "Segoe UI", sans-serif; font-weight: 650; }
    input {
        border-color: #d7d9e0;
        background: #fff;
    }
    input:focus { border-color: var(--olive); }
    .password-toggle:hover { background: #f0f2ff; color: var(--forest); }
    .password-checklist { background: #f8f9fc; }
    .password-checklist li.is-met { color: #304ccf; }
    input:focus-visible, button:focus-visible, a:focus-visible {
        outline-color: #4158f5;
    }
    button { border-radius: 9px; background: #4158f5; }
    button:hover { background: #3048df; }
    .message { border-color: #f0c5bd; background: #fff5f2; color: #713e31; }
    a { color: #3048df; }
    .privacy { color: #a7aab4; }
</style>
""",
)


def _auth_page(
    mode: str,
    message: str = "",
    username: str = "",
    status_code: int = 200,
) -> HTMLResponse:
    is_register = mode == "register"
    title = "Create your account" if is_register else "Welcome back"
    description = (
        "Create a local account to use the DeepTrace analysis workspace."
        if is_register
        else "Sign in to continue to the DeepTrace analysis workspace."
    )
    fields = """
        <label for="username">Username</label>
        <input id="username" name="username" type="text" required
            minlength="3" maxlength="32" autocomplete="username"
            pattern="[A-Za-z0-9][A-Za-z0-9_.\\x2d]{{2,31}}" value="{username}">
        <label for="password">Password</label>
        <div class="password-field">
            <input id="password" name="password" type="password" required
                minlength="6" maxlength="128" autocomplete="{password_autocomplete}"
                {password_pattern} title="{password_title}">
            <button class="password-toggle" type="button" data-password-toggle="password"
                aria-controls="password" aria-pressed="false">Show</button>
        </div>
    """.format(
        username=escape(username, quote=True),
        password_autocomplete="new-password" if is_register else "current-password",
        password_pattern=(
            'pattern="(?=.*[A-Za-z])(?=.*[0-9])(?=[A-Za-z0-9]*[^A-Za-z0-9\\s][A-Za-z0-9]*$)[^\\s]{6,128}" aria-describedby="password-checklist password-status"'
            if is_register
            else ""
        ),
        password_title=(
            "Use 6-128 characters, including letters, numbers, and exactly one special character."
            if is_register
            else ""
        ),
    )
    if is_register:
        fields += """
            <ul id="password-checklist" class="password-checklist"
                aria-label="Password requirements">
                <li data-password-rule="length">
                    <span class="requirement-indicator" aria-hidden="true">○</span>
                    <span>6–128 characters</span>
                </li>
                <li data-password-rule="letter">
                    <span class="requirement-indicator" aria-hidden="true">○</span>
                    <span>At least one letter</span>
                </li>
                <li data-password-rule="number">
                    <span class="requirement-indicator" aria-hidden="true">○</span>
                    <span>At least one number</span>
                </li>
                <li data-password-rule="special">
                    <span class="requirement-indicator" aria-hidden="true">○</span>
                    <span>Exactly one special character</span>
                </li>
                <li data-password-rule="characters">
                    <span class="requirement-indicator" aria-hidden="true">○</span>
                    <span>No spaces or non-ASCII characters</span>
                </li>
            </ul>
            <p id="password-status" class="password-status" role="status" aria-live="polite">
                0 of 5 requirements met.
            </p>
            <label for="password_confirmation">Confirm password</label>
            <div class="password-field">
                <input id="password_confirmation" name="password_confirmation"
                    type="password" required minlength="6" maxlength="128"
                    autocomplete="new-password"
                    pattern="(?=.*[A-Za-z])(?=.*[0-9])(?=[A-Za-z0-9]*[^A-Za-z0-9\\s][A-Za-z0-9]*$)[^\\s]{6,128}"
                    title="Use 6-128 characters, including letters, numbers, and exactly one special character.">
                <button class="password-toggle" type="button"
                    data-password-toggle="password_confirmation"
                    aria-controls="password_confirmation" aria-pressed="false">Show</button>
            </div>
        """

    message_html = (
        f'<p class="message" role="alert">{escape(message)}</p>' if message else ""
    )
    target = "/register" if is_register else "/login"
    switch = (
        'Already registered? <a href="/">Sign in</a>'
        if is_register
        else 'New to DeepTrace? <a href="/register">Create an account</a>'
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="color-scheme" content="light">
    <title>{title} · DeepTrace</title>
    {_AUTH_CSS}
</head>
<body>
    <main class="page">
        <a class="brand" href="/">Deep<span>Trace.</span></a>
        <section class="card" aria-labelledby="page-title">
            <h1 id="page-title">{title}</h1>
            <p class="intro">{description}</p>
            <form method="post" action="{target}">
                {fields}
                <button type="submit">{'Create account' if is_register else 'Sign in'}</button>
            </form>
            {message_html}
            <p class="switch">{switch}</p>
        </section>
        <p class="privacy">Local accounts are stored on this device. Passwords are never stored as plain text.</p>
    </main>
<script>
    document.querySelectorAll("[data-password-toggle]").forEach((toggle) => {{
        toggle.addEventListener("click", () => {{
            const input = document.getElementById(toggle.dataset.passwordToggle);
            const showing = input.type === "password";
            input.type = showing ? "text" : "password";
            toggle.textContent = showing ? "Hide" : "Show";
            toggle.setAttribute("aria-pressed", String(showing));
        }});
    }});
    const registrationPassword = document.getElementById("password-checklist")
        ? document.getElementById("password")
        : null;
    if (registrationPassword) {{
        const requirementItems = document.querySelectorAll("[data-password-rule]");
        const status = document.getElementById("password-status");
        const updatePasswordChecklist = () => {{
            const value = registrationPassword.value;
            const characters = Array.from(value);
            const specialCount = characters.filter((character) => {{
                const code = character.codePointAt(0);
                return (code >= 33 && code <= 47)
                    || (code >= 58 && code <= 64)
                    || (code >= 91 && code <= 96)
                    || (code >= 123 && code <= 126);
            }}).length;
            const checks = {{
                length: value.length >= 6 && value.length <= 128,
                letter: /[A-Za-z]/.test(value),
                number: /[0-9]/.test(value),
                special: specialCount === 1,
                characters: value.length > 0 && !/[^\\x00-\\x7F]|\\s/.test(value),
            }};
            let metCount = 0;
            requirementItems.forEach((item) => {{
                const met = checks[item.dataset.passwordRule];
                const label = item.lastElementChild.textContent;
                item.classList.toggle("is-met", met);
                item.setAttribute("aria-label", label + (met ? ": met" : ": not met"));
                item.querySelector(".requirement-indicator").textContent = met ? "✓" : "○";
                if (met) metCount += 1;
            }});
            status.textContent = metCount + " of 5 requirements met.";
        }};
        registrationPassword.addEventListener("input", updatePasswordChecklist);
        updatePasswordChecklist();
    }}
</script>
</body>
</html>"""
    return HTMLResponse(
        page,
        status_code=status_code,
        headers={"Cache-Control": "no-store"},
    )


def create_auth_app(
    blocks: gr.Blocks,
    store: AuthStore | None = None,
    css: str | None = None,
    theme: Any = None,
) -> FastAPI:
    auth_store = store or AuthStore()
    web_app = FastAPI(title="DeepTrace local application")

    def authenticated_user(request: Request) -> str | None:
        return auth_store.get_session_user(request.cookies.get(SESSION_COOKIE))

    def issue_session(username: str) -> RedirectResponse:
        response = RedirectResponse(url="/app", status_code=303)
        response.set_cookie(
            key=SESSION_COOKIE,
            value=auth_store.create_session(username),
            max_age=auth_store.session_ttl_seconds,
            httponly=True,
            samesite="lax",
            secure=auth_store.secure_cookie,
            path="/",
        )
        return response

    @web_app.get("/", response_class=HTMLResponse)
    def login_page(request: Request) -> Response:
        if authenticated_user(request):
            return RedirectResponse(url="/app", status_code=303)
        return _auth_page("login")

    @web_app.post("/login")
    def login(
        username: str = Form(...),
        password: str = Form(...),
    ) -> Response:
        authenticated_username = auth_store.authenticate(username, password)
        if authenticated_username is None:
            return _auth_page(
                "login",
                "Username or password is incorrect.",
                username=username,
                status_code=401,
            )
        return issue_session(authenticated_username)

    @web_app.get("/register", response_class=HTMLResponse)
    def registration_page(request: Request) -> Response:
        if authenticated_user(request):
            return RedirectResponse(url="/app", status_code=303)
        return _auth_page("register")

    @web_app.post("/register")
    def register(
        username: str = Form(...),
        password: str = Form(...),
        password_confirmation: str = Form(...),
    ) -> Response:
        try:
            created_username = auth_store.register(
                username,
                password,
                password_confirmation,
            )
        except DuplicateAccountError as exc:
            return _auth_page(
                "register",
                str(exc),
                username=username,
                status_code=409,
            )
        except RegistrationError as exc:
            return _auth_page(
                "register",
                str(exc),
                username=username,
                status_code=400,
            )
        return issue_session(created_username)

    @web_app.post("/logout")
    def logout(request: Request) -> RedirectResponse:
        auth_store.revoke_session(request.cookies.get(SESSION_COOKIE))
        response = RedirectResponse(url="/", status_code=303)
        response.delete_cookie(
            key=SESSION_COOKIE,
            path="/",
            httponly=True,
            samesite="lax",
            secure=auth_store.secure_cookie,
        )
        return response

    gr.mount_gradio_app(
        web_app,
        blocks,
        path="/app",
        auth_dependency=authenticated_user,
        css=css,
        theme=theme,
        show_error=False,
    )
    return web_app
