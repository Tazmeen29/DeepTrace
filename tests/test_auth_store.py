import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import gradio as gr
from fastapi.testclient import TestClient

from auth_store import AuthStore, DuplicateAccountError, RegistrationError
from auth_web import create_auth_app


VALID_PASSWORD = "alongpassword1!"


class AuthStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = AuthStore(
            Path(self.temp_dir.name) / "accounts.sqlite3",
            session_ttl_seconds=60,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_registration_hashes_password_and_authenticates(self) -> None:
        username = self.store.register(
            "  DeepTrace.User ",
            VALID_PASSWORD,
            VALID_PASSWORD,
        )

        self.assertEqual(username, "deeptrace.user")
        self.assertEqual(
            self.store.authenticate(username, VALID_PASSWORD),
            username,
        )
        self.assertIsNone(self.store.authenticate(username, "wrong-password"))

    def test_duplicate_account_is_rejected_without_case_sensitivity(self) -> None:
        self.store.register("Researcher", VALID_PASSWORD, VALID_PASSWORD)

        with self.assertRaises(DuplicateAccountError):
            self.store.register(
                "researcher",
                "anotherpassword2!",
                "anotherpassword2!",
            )

    def test_registration_validates_password_confirmation_and_username(self) -> None:
        with self.assertRaisesRegex(RegistrationError, "do not match"):
            self.store.register("researcher", "a-long-and-safe-password", "not-the-same")
        with self.assertRaises(RegistrationError):
            self.store.register("!", "a-long-and-safe-password", "a-long-and-safe-password")
        with self.assertRaisesRegex(RegistrationError, "at least 6"):
            self.store.register("researcher", "a1!xy", "a1!xy")

    def test_six_character_password_is_accepted(self) -> None:
        self.store.register("researcher", "a1!xyz", "a1!xyz")
        self.assertEqual(self.store.authenticate("researcher", "a1!xyz"), "researcher")

    def test_password_requires_letters_numbers_and_exactly_one_special(self) -> None:
        invalid_passwords = (
            ("abcdef!", "at least one number"),
            ("123456!", "at least one letter"),
            ("abcdef1", "exactly one special character"),
            ("abc123!!", "exactly one special character"),
            ("abc 12!", "no spaces"),
            ("abc12é!", "no spaces"),
        )
        for password, message in invalid_passwords:
            with self.subTest(password=password):
                with self.assertRaisesRegex(RegistrationError, message):
                    self.store.register("researcher", password, password)

    def test_sessions_expire_and_can_be_revoked(self) -> None:
        username = self.store.register(
            "researcher",
            VALID_PASSWORD,
            VALID_PASSWORD,
        )
        token = self.store.create_session(username)

        self.assertEqual(self.store.get_session_user(token), username)
        self.assertIsNone(self.store.get_session_user(token, now=2**31))

        self.store.revoke_session(token)
        self.assertIsNone(self.store.get_session_user(token))
        self.assertIsNone(self.store.get_session_user(token + "é"))

    def test_session_token_is_not_stored_in_plaintext(self) -> None:
        username = self.store.register(
            "researcher",
            VALID_PASSWORD,
            VALID_PASSWORD,
        )
        token = self.store.create_session(username)

        import sqlite3

        with closing(sqlite3.connect(self.store.db_path)) as connection:
            stored_token = connection.execute(
                "SELECT token_hash FROM sessions"
            ).fetchone()[0]
            stored_password = connection.execute(
                "SELECT password_hash FROM users"
            ).fetchone()[0]

        self.assertNotEqual(stored_token, token)
        self.assertNotIn(VALID_PASSWORD, stored_password)
        self.assertTrue(stored_password.startswith("scrypt$"))

    def test_registration_login_and_logout_protect_gradio(self) -> None:
        with gr.Blocks() as protected_blocks:
            gr.Markdown("Protected analysis workspace")
        with TestClient(create_auth_app(protected_blocks, self.store)) as client:
            self.assertEqual(client.get("/app/").status_code, 401)
            self.assertEqual(client.get("/app/config").status_code, 401)
            registration = client.post(
                "/register",
                data={
                    "username": "reviewer",
                    "password": VALID_PASSWORD,
                    "password_confirmation": VALID_PASSWORD,
                },
                follow_redirects=False,
            )
            self.assertEqual(registration.status_code, 303)
            self.assertEqual(registration.headers["location"], "/app")
            self.assertIn("httponly", registration.headers["set-cookie"].lower())
            self.assertEqual(client.get("/app/").status_code, 200)
            self.assertEqual(client.get("/app/config").status_code, 200)

            logout = client.post("/logout", follow_redirects=False)
            self.assertEqual(logout.status_code, 303)
            self.assertEqual(client.get("/app/").status_code, 401)

            login = client.post(
                "/login",
                data={
                    "username": "reviewer",
                    "password": VALID_PASSWORD,
                },
                follow_redirects=False,
            )
            self.assertEqual(login.status_code, 303)
            self.assertEqual(client.get("/app/config").status_code, 200)

    def test_invalid_login_returns_friendly_message(self) -> None:
        with TestClient(create_auth_app(gr.Blocks(), self.store)) as client:
            response = client.post(
                "/login",
                data={"username": "unknown", "password": "incorrect"},
            )

        self.assertEqual(response.status_code, 401)
        self.assertIn("Username or password is incorrect.", response.text)

    def test_registration_form_has_password_visibility_and_policy_hints(self) -> None:
        with TestClient(create_auth_app(gr.Blocks(), self.store)) as client:
            response = client.get("/register")

        self.assertEqual(response.status_code, 200)
        self.assertIn('data-password-toggle="password"', response.text)
        self.assertIn('data-password-toggle="password_confirmation"', response.text)
        self.assertIn("6-128 characters", response.text)
        self.assertIn("exactly one special character", response.text)
        self.assertIn('input.type = showing ? "text" : "password"', response.text)
        for requirement in ("length", "letter", "number", "special", "characters"):
            self.assertIn(f'data-password-rule="{requirement}"', response.text)
        self.assertIn(
            'registrationPassword.addEventListener("input", updatePasswordChecklist)',
            response.text,
        )
        self.assertIn('id="password-status"', response.text)


if __name__ == "__main__":
    unittest.main()
