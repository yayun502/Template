'''
## Usage
# main.py
from __future__ import annotations

from pathlib import Path

import requests
import yaml

from secrets_manager import decrypt_secret


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def call_service(
    base_url: str,
    pat: str,
    endpoint: str,
) -> requests.Response:
    response = requests.get(
        f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}",
        headers={
            "Authorization": f"Bearer {pat}",
            "Accept": "application/json",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response


def main() -> None:
    config = load_config("config.yaml")

    service_a_config = config["services"]["service_a"]
    service_b_config = config["services"]["service_b"]

    service_a_pat = decrypt_secret(
        service_a_config["encrypted_pat"],
        "service_a_pat",
    )

    service_b_pat = decrypt_secret(
        service_b_config["encrypted_pat"],
        "service_b_pat",
    )

    result_a = call_service(
        service_a_config["base_url"],
        service_a_pat,
        "/resource",
    )

    result_b = call_service(
        service_b_config["base_url"],
        service_b_pat,
        "/resource",
    )

    print("Service A status:", result_a.status_code)
    print("Service B status:", result_b.status_code)


if __name__ == "__main__":
    main()
'''
# secrets_manager.py
from __future__ import annotations

import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class SecretError(RuntimeError):
    pass


def load_master_key() -> bytes:
    encoded_key = os.environ.get("APP_MASTER_KEY")

    if not encoded_key:
        raise SecretError("未提供 APP_MASTER_KEY")

    try:
        key = base64.urlsafe_b64decode(encoded_key)
    except Exception as exc:
        raise SecretError("APP_MASTER_KEY 無法解析") from exc

    if len(key) not in (16, 24, 32):
        raise SecretError("APP_MASTER_KEY 長度不正確")

    return key


def decrypt_secret(encrypted_value: str, secret_name: str) -> str:
    try:
        payload = base64.urlsafe_b64decode(encrypted_value)
    except Exception as exc:
        raise SecretError(f"{secret_name} 的加密內容格式錯誤") from exc

    if len(payload) <= 12:
        raise SecretError(f"{secret_name} 的加密內容過短")

    nonce = payload[:12]
    ciphertext = payload[12:]

    key = load_master_key()
    aesgcm = AESGCM(key)

    try:
        plaintext = aesgcm.decrypt(
            nonce,
            ciphertext,
            secret_name.encode("utf-8"),
        )
    except Exception as exc:
        raise SecretError(
            f"{secret_name} 解密失敗，可能是金鑰錯誤或內容遭竄改"
        ) from exc

    return plaintext.decode("utf-8")
