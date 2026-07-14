'''
# generate_key.py
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

key = AESGCM.generate_key(bit_length=256)
print(base64.urlsafe_b64encode(key).decode("ascii"))
'''
# encrypt_secret.py
from __future__ import annotations

import argparse
import base64
import getpass
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def load_master_key() -> bytes:
    encoded_key = os.environ.get("APP_MASTER_KEY")

    if not encoded_key:
        raise RuntimeError(
            "找不到 APP_MASTER_KEY，請從安全來源提供主金鑰。"
        )

    try:
        key = base64.urlsafe_b64decode(encoded_key)
    except Exception as exc:
        raise RuntimeError("APP_MASTER_KEY 格式錯誤") from exc

    if len(key) not in (16, 24, 32):
        raise RuntimeError("AES 金鑰長度必須為 128、192 或 256 bits")

    return key


def encrypt_secret(secret: str, secret_name: str) -> str:
    key = load_master_key()
    aesgcm = AESGCM(key)

    # 每次加密都必須產生新的 nonce。
    nonce = os.urandom(12)

    # AAD 不加密，但會納入完整性驗證。
    aad = secret_name.encode("utf-8")

    ciphertext = aesgcm.encrypt(
        nonce,
        secret.encode("utf-8"),
        aad,
    )

    payload = nonce + ciphertext
    return base64.urlsafe_b64encode(payload).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        required=True,
        help="Secret 名稱，例如 service_a_pat",
    )
    args = parser.parse_args()

    secret = getpass.getpass("請輸入 PAT：")
    encrypted = encrypt_secret(secret, args.name)

    print("\n請將以下內容放進 config：")
    print(encrypted)


if __name__ == "__main__":
    main()
