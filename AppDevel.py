import secrets

def generate_secret_key():
    # Generates a 32-character hexadecimal string.
    return secrets.token_hex(16)

if __name__ == '__main__':
    secret_key = generate_secret_key()
    print("Your Flask secret key is:", secret_key)
