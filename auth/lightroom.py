"""
Adobe Lightroom OAuth2 — authorization code flow with PKCE.

First-time login (one-time setup):
    uv run python auth/lightroom.py

Chrome: click 'Advanced' → 'Proceed to localhost' when the security warning appears.
Safari: after the redirect fails, press Cmd+L then Cmd+C — detected automatically.
"""

import base64
import datetime
import hashlib
import http.server
import json
import os
import secrets
import ssl
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

AUTHORIZE_URL = "https://ims-na1.adobelogin.com/ims/authorize/v2"
TOKEN_URL = "https://ims-na1.adobelogin.com/ims/token/v3"
SCOPES = "openid,AdobeID,lr_partner_apis,lr_partner_rendition_apis,offline_access"


def _make_ssl_context() -> tuple[ssl.SSLContext, str, str]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.UTC))
        .not_valid_after(datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
        .sign(key, hashes.SHA256())
    )
    tmp = tempfile.mkdtemp()
    cert_path = os.path.join(tmp, "cert.pem")
    key_path = os.path.join(tmp, "key.pem")
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(key_path, "wb") as f:
        f.write(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            )
        )
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_path, key_path)
    return ctx, cert_path, key_path, tmp


def _pkce() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def _clipboard() -> str:
    try:
        return subprocess.check_output(["pbpaste"], text=True).strip()
    except Exception:
        return ""


def _exchange_code(code: str, verifier: str, client_id: str, client_secret: str, redirect_uri: str) -> dict:
    data = urllib.parse.urlencode(
        {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
        }
    ).encode()
    req = urllib.request.Request(TOKEN_URL, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def refresh_access_token(client_id: str, client_secret: str, refresh_token: str) -> str:
    data = urllib.parse.urlencode(
        {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
        }
    ).encode()
    req = urllib.request.Request(TOKEN_URL, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["access_token"]


def _write_refresh_token(token: str) -> None:
    env_path = Path(".env")
    if not env_path.exists():
        env_path.write_text(f"LIGHTROOM_REFRESH_TOKEN={token}\n")
        return
    lines = env_path.read_text().splitlines(keepends=True)
    updated = False
    for i, line in enumerate(lines):
        if line.startswith("LIGHTROOM_REFRESH_TOKEN="):
            lines[i] = f"LIGHTROOM_REFRESH_TOKEN={token}\n"
            updated = True
            break
    if not updated:
        lines.append(f"LIGHTROOM_REFRESH_TOKEN={token}\n")
    env_path.write_text("".join(lines))


def login(client_id: str, client_secret: str, redirect_uri: str) -> str:
    """OAuth2 PKCE flow. Tries https server (Chrome), falls back to clipboard (Safari)."""
    parsed = urllib.parse.urlparse(redirect_uri)
    port = parsed.port or 443
    callback_prefix = redirect_uri.split("?")[0]
    verifier, challenge = _pkce()
    state = secrets.token_urlsafe(16)

    params = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": SCOPES,
            "response_type": "code",
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )
    auth_url = f"{AUTHORIZE_URL}?{params}"

    # ── https server (Chrome path) ────────────────────────────────────────────
    server_result: dict = {}
    server_ready = threading.Event()
    server_error: list = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            if "code" in qs:
                server_result["code"] = qs["code"][0]
                self.wfile.write(b"<h2>Authenticated! You can close this tab.</h2>")
            else:
                server_result["error"] = qs.get("error", ["unknown"])[0]
                self.wfile.write(b"<h2>Authentication failed.</h2>")

        def log_message(self, *_):
            pass

    try:
        ssl_ctx, cert_path, key_path, tmp_dir = _make_ssl_context()
        httpd = http.server.HTTPServer(("localhost", port), Handler)
        httpd.socket = ssl_ctx.wrap_socket(httpd.socket, server_side=True)

        def serve():
            server_ready.set()
            while not server_result:
                try:
                    httpd.handle_request()
                except ssl.SSLError:
                    pass

        t = threading.Thread(target=serve, daemon=True)
        t.start()
        server_ready.wait()
    except Exception as e:
        server_error.append(str(e))
        cert_path = key_path = tmp_dir = None

    print("\nOpening Adobe login in your browser...")
    try:
        import webbrowser

        webbrowser.open(auth_url)
    except Exception:
        print(f"Open this URL manually:\n  {auth_url}\n")

    if not server_error:
        print("Chrome: click 'Advanced' → 'Proceed to localhost' if a security warning appears.")
    print("Safari: after the redirect fails, press Cmd+L then Cmd+C — detected automatically.\n")

    # ── clipboard fallback (Safari path) — runs in parallel ──────────────────
    clipboard_result: dict = {}
    seen_clipboard = _clipboard()

    def watch_clipboard():
        deadline = time.time() + 300
        while time.time() < deadline and not server_result and not clipboard_result:
            text = _clipboard()
            if text != seen_clipboard and text.startswith(callback_prefix) and "code=" in text:
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(text).query)
                if "code" in qs:
                    clipboard_result["code"] = qs["code"][0]
                    print("Callback URL detected from clipboard.")
            time.sleep(1)

    clip_t = threading.Thread(target=watch_clipboard, daemon=True)
    clip_t.start()

    # Wait for whichever arrives first
    deadline = time.time() + 300
    while time.time() < deadline:
        if server_result or clipboard_result:
            break
        time.sleep(0.5)

    # Cleanup cert files and temp dir
    for f in (cert_path, key_path):
        if f:
            try:
                os.unlink(f)
            except Exception:
                pass
    if tmp_dir:
        try:
            os.rmdir(tmp_dir)
        except Exception:
            pass

    result = server_result or clipboard_result
    if "error" in result:
        raise RuntimeError(f"OAuth error: {result['error']}")
    if "code" not in result:
        raise RuntimeError("Timed out waiting for OAuth callback (5 min).")

    tokens = _exchange_code(result["code"], verifier, client_id, client_secret, redirect_uri)
    refresh_token = tokens["refresh_token"]
    _write_refresh_token(refresh_token)
    print("Refresh token saved to .env\n")
    return refresh_token


def get_access_token() -> str:
    """Return a valid access token, running login flow if no token exists."""
    from dotenv import load_dotenv

    load_dotenv()
    client_id = os.environ.get("LIGHTROOM_CLIENT_ID", "")
    client_secret = os.environ.get("LIGHTROOM_CLIENT_SECRET", "")
    redirect_uri = os.environ.get("LIGHTROOM_REDIRECT_URI", "https://localhost:8765/callback")
    refresh_token = os.environ.get("LIGHTROOM_REFRESH_TOKEN", "")

    if not client_id or not client_secret:
        raise RuntimeError("LIGHTROOM_CLIENT_ID and LIGHTROOM_CLIENT_SECRET must be set in .env")

    if not refresh_token:
        refresh_token = login(client_id, client_secret, redirect_uri)
        return refresh_access_token(client_id, client_secret, refresh_token)

    try:
        return refresh_access_token(client_id, client_secret, refresh_token)
    except Exception as e:
        _write_refresh_token("")
        raise RuntimeError(f"Refresh token invalid ({e}). It has been cleared from .env.\nRe-run authentication: uv run python auth/lightroom.py") from None


if __name__ == "__main__":
    token = get_access_token()
    print("Authenticated with Lightroom. You can now run: uv run python main.py")
