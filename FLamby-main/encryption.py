from phe import paillier

# Generate global keypair for simplicity (in real settings, you'd separate parties)
pubkey, privkey = paillier.generate_paillier_keypair()

def encrypt_matrix(matrix, pubkey):
    return [[pubkey.encrypt(float(val)) for val in row] for row in matrix]

def decrypt_vector(vec, privkey):
    return [privkey.decrypt(val) for val in vec]
