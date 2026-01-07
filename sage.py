from sage.crypto.sbox import SBox
from sage.crypto.boolean_function import BooleanFunction

def cls(byte, shift):
    b = (byte & 0x0F) << shift
    return ((b | (b >> 4)) & 0x0F)

def generate_sbox(T, r, S, use_xor=True):
    sbox = []
    for x in range(256):
        y = x
        for i in range(T):
            X1 = (y >> 4) & 0x0F
            X2 = y & 0x0F
            Z  = (X1 + cls(X2, r[i])) & 0x0F
            if use_xor:
                Y2 = (X2 ^ cls(Z, S[i])) & 0x0F
            else:
                Y2 = (X2 + cls(Z, S[i])) & 0x0F
            y = (Z << 4) | Y2
        sbox.append(y)
    return sbox

def compute_lambda_pi(sbox_obj):
    min_degree = 8
    for b in range(1, 256):
        bf = sbox_obj.component_function(b)
        deg = bf.algebraic_degree()
        if deg < min_degree:
            min_degree = deg
            if min_degree == 0:
                break
    return min_degree

def compute_delta_pi(sbox_obj):
    LAT = sbox_obj.linear_approximation_table()
    max_abs = 0
    for a in range(1, 256):
        for b in range(1, 256):
            v = abs(LAT[a][b])
            if v > max_abs:
                max_abs = v
    return 2 * float(max_abs) / 256.0

def check(T, r, S, p_pi=None, lambda_pi=None, delta_pi=None):
    sbox_list = generate_sbox(T, r, S)
    sbox = SBox(sbox_list)

    if not sbox.is_permutation():
        print(f"T={T} r={r} S={S} → Не биективна")
        return

    du = sbox.differential_uniformity()
    sage_prob = du / 256.0

    if p_pi is not None:
        match = abs(p_pi - sage_prob) < 1e-9
        result = "Совпадение" if match else "Расхождение"
        print(f"{result} | Sage p_pi: {sage_prob:.7f}, Ваша: {p_pi:.7f}")
    else:
        print(f"T={T} r={r} S={S} → DU={du} (p_pi={sage_prob:.7f})")

    lp = compute_lambda_pi(sbox)
    if lambda_pi is not None:
        match = abs(lambda_pi - lp) < 1e-9
        result = "Совпадение" if match else "Расхождение"
        print(f"{result} | Sage lambda_pi: {lp}, Ваша: {lambda_pi}")
    else:
        print(f"(lambda_pi={lp})")

    dp = compute_delta_pi(sbox)
    if delta_pi is not None:
        match = abs(delta_pi - dp) < 1e-9
        result = "Совпадение" if match else "Расхождение"
        print(f"{result} | Sage delta_pi: {dp:.7f}, Ваша: {delta_pi:.7f}")
    else:
        print(f"(delta_pi={dp:.7f})")




    
    
T = 9
r = [2, 0, 3, 3, 2, 1, 3, 3, 2 ]
S = [1, 2, 0, 3, 0, 0, 0, 0, 0 ]
p_pi = 0.0312500
lambda_pi = 7
delta_pi = 0.2343750
check(T, r, S, p_pi, lambda_pi, delta_pi)

