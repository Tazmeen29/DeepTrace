import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import string
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


_USERNAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,31}$")
_PASSWORD_SPECIAL_CHARACTERS = frozenset(string.punctuation)
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1
_SESSION_COOKIE = "deeptrace_session"


class RegistrationError(ValueError):
    pass


class DuplicateAccountError(RegistrationError):
    pass


class AuthStore:
    def __init__(
        self,
        db_path: str | Path | None = None,
        session_ttl_seconds: int | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parent
        configured_path = db_path or os.getenv("DEEPTRACE_DB_PATH")
        path = Path(configured_path) if configured_path else Path("data/deeptrace.sqlite3")
        self.db_path = path if path.is_absolute() else project_root / path

        if session_ttl_seconds is None:
            session_ttl_seconds = int(os.getenv("DEEPTRACE_SESSION_TTL_HOURS", "12")) * 60 * 60
        if session_ttl_seconds <= 0:
            raise ValueError("Session lifetime must be greater than zero.")
        self.session_ttl_seconds = session_ttl_seconds
        self.secure_cookie = os.getenv("DEEPTRACE_COOKIE_SECURE", "false").lower() in {
            "1",
            "true",
            "yes",
        }

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
                    expires_at INTEGER NOT NULL,
                    created_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS sessions_expiry_idx
                    ON sessions(expires_at);
                """
            )

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.db_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _normalize_username(username: str) -> str:
        normalized = username.strip().lower()
        if not _USERNAME_PATTERN.fullmatch(normalized):
            raise RegistrationError(
                "Use 3–32 letters, numbers, dots, underscores, or hyphens; "
                "start with a letter or number."
            )
        return normalized

    @staticmethod
    def _hash_password(password: str) -> str:
        salt = secrets.token_bytes(16)
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=_SCRYPT_N,
            r=_SCRYPT_R,
            p=_SCRYPT_P,
            maxmem=64 * 1024 * 1024,
        )
        return (
            f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}"
            f"${salt.hex()}${digest.hex()}"
        )

    @staticmethod
    def _password_matches(password: str, stored_hash: str) -> bool:
        try:
            algorithm, n, r, p, salt_hex, digest_hex = stored_hash.split("$")
            if algorithm != "scrypt":
                return False
            expected = bytes.fromhex(digest_hex)
            actual = hashlib.scrypt(
                password.encode("utf-8"),
                salt=bytes.fromhex(salt_hex),
                n=int(n),
                r=int(r),
                p=int(p),
                maxmem=64 * 1024 * 1024,
            )
        except (ValueError, TypeError):
            return False
        return hmac.compare_digest(actual, expected)

    def register(
        self,
        username: str,
        password: str,
        password_confirmation: str,
    ) -> str:
        normalized = self._normalize_username(username)
        if password != password_confirmation:
            raise RegistrationError("Passwords do not match.")
        if len(password) < 6:
            raise RegistrationError("Use a password with at least 6 characters.")
        if len(password) > 128:
            raise RegistrationError("Passwords must be 128 characters or fewer.")
        if not any(character.isascii() and character.isalpha() for character in password):
            raise RegistrationError("Include at least one letter.")
        if not any(character.isascii() and character.isdigit() for character in password):
            raise RegistrationError("Include at least one number.")
        special_count = sum(
            character in _PASSWORD_SPECIAL_CHARACTERS for character in password
        )
        if special_count != 1 or any(
            not character.isascii() or character.isspace() for character in password
        ):
            raise RegistrationError(
                "Use exactly one special character and no spaces."
            )

        password_hash = self._hash_password(password)
        created_at = int(time.time())
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO users (username, password_hash, created_at) "
                    "VALUES (?, ?, ?)",
                    (normalized, password_hash, created_at),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateAccountError(
                "That username is already registered. Please sign in instead."
            ) from exc
        return normalized

    def authenticate(self, username: str, password: str) -> str | None:
        normalized = username.strip().lower()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT username, password_hash FROM users WHERE username = ?",
                (normalized,),
            ).fetchone()
        if row is None or not self._password_matches(password, row["password_hash"]):
            return None
        return str(row["username"])

    def create_session(self, username: str) -> str:
        token = secrets.token_urlsafe(32)
        now = int(time.time())
        with self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            connection.execute(
                "INSERT INTO sessions (token_hash, username, expires_at, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    hashlib.sha256(token.encode("ascii")).hexdigest(),
                    username,
                    now + self.session_ttl_seconds,
                    now,
                ),
            )
        return token

    @staticmethod
    def _session_token_hash(token: str | None) -> str | None:
        if not token or len(token) > 256:
            return None
        try:
            token_bytes = token.encode("ascii")
        except UnicodeEncodeError:
            return None
        return hashlib.sha256(token_bytes).hexdigest()

    def get_session_user(self, token: str | None, now: int | None = None) -> str | None:
        token_hash = self._session_token_hash(token)
        if token_hash is None:
            return None
        current_time = int(time.time()) if now is None else now
        with self._connect() as connection:
            row = connection.execute(
                "SELECT username FROM sessions "
                "WHERE token_hash = ? AND expires_at > ?",
                (token_hash, current_time),
            ).fetchone()
        return str(row["username"]) if row else None

    def revoke_session(self, token: str | None) -> None:
        token_hash = self._session_token_hash(token)
        if token_hash is None:
            return
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM sessions WHERE token_hash = ?",
                (token_hash,),
            )


SESSION_COOKIE = _SESSION_COOKIE
