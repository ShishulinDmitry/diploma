#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>


static inline uint8_t cls4(uint8_t v, uint8_t shift) {
    shift &= 3; // mod 4
    uint8_t x = (v << shift) & 0x0F;
    uint8_t y = (v >> (4 - shift)) & 0x0F;
    return (x | y) & 0x0F;
}

static inline uint8_t A_single(uint8_t X, uint8_t r, uint8_t S) {
    uint8_t X1 = (X >> 4) & 0x0F;
    uint8_t X2 = X & 0x0F;

    uint8_t Z  = (X1 + cls4(X2, r)) & 0x0F;
    uint8_t Y2 = (X2 ^ cls4(Z, S)) & 0x0F; 

    return (uint8_t)((Z << 4) | Y2);
}

static uint8_t A_rounds(uint8_t X, const uint8_t *r, const uint8_t *S, uint8_t T) {
    uint8_t Y = X;
    for (uint8_t i = 0; i < T; ++i) {
        Y = A_single(Y, r[i], S[i]);
    }
    return Y;
}

static void build_sbox(uint8_t sbox[256], const uint8_t *r, const uint8_t *S, uint8_t T) {
    for (int x = 0; x < 256; ++x) {
        sbox[x] = A_rounds((uint8_t)x, r, S, T);
    }
}


static int is_permutation(const uint8_t sbox[256]) {
    uint8_t used[256];
    memset(used, 0, sizeof(used));
    for (int x = 0; x < 256; ++x) {
        if (used[sbox[x]]) {
            return 0; 
        }
        used[sbox[x]] = 1;
    }
    return 1; 
}


static float compute_p_pi(const uint8_t Y[256]) {
    int   counter[256][256] = {0};
    float max_prob = 0.0f;

    for (int x = 0; x < 256; x++) {
        for (int a = 1; a < 256; a++) {
            int x_prime = x ^ a;
            int eps     = Y[x] ^ Y[x_prime];
            if (eps == 0) continue;       
            counter[a][eps]++;
        }
    }

    for (int a = 1; a < 256; a++) {
        int total = 0;
        for (int eps = 1; eps < 256; eps++) total += counter[a][eps];
        if (total == 0) continue;

        for (int eps = 1; eps < 256; eps++) {
            float prob = (float)counter[a][eps] / (float)total;
            if (prob > max_prob) max_prob = prob;
        }
    }
    return max_prob;
}

static int anf_degree(uint8_t *F) {
    uint8_t coeffs[256];
    memcpy(coeffs, F, 256);

    for (int i = 0; i < 8; i++) {
        for (int mask = 0; mask < 256; mask++) {
            if (mask & (1 << i)) {
                coeffs[mask] ^= coeffs[mask ^ (1 << i)];
            }
        }
    }

    int deg = 0;
    for (int mask = 1; mask < 256; mask++) {
        if (coeffs[mask]) {
            int d = __builtin_popcount(mask);
            if (d > deg) deg = d;
        }
    }
    return deg;
}

static int compute_lambda(uint8_t *Y) {
    int lambda = 8;
    uint8_t Yinv[256];

    for (int x = 0; x < 256; x++) {
        Yinv[Y[x]] = x;
    }

    for (int a = 1; a < 256; a++) {
        uint8_t f[256], g[256];

        for (int x = 0; x < 256; x++) {
            f[x] = __builtin_parity(a & Y[x]);
        }
        for (int x = 0; x < 256; x++) {
            g[x] = __builtin_parity(a & Yinv[x]);
        }

        int deg_f   = anf_degree(f);
        int deg_g   = anf_degree(g);
        int max_deg = (deg_f > deg_g ? deg_f : deg_g);

        if (max_deg < lambda) lambda = max_deg;
    }
    return lambda;
}


static void fwt_int_256(int f[256]) {
    for (int len = 1; len < 256; len <<= 1) {
        for (int start = 0; start < 256; start += (len << 1)) {
            for (int i = 0; i < len; i++) {
                int a = f[start + i];
                int b = f[start + i + len];
                f[start + i]       = a + b;
                f[start + i + len] = a - b;
            }
        }
    }
}

static float compute_delta(const uint8_t Y[256]) {
    const float norm = 1.0f / 256.0f;  
    float max_abs_delta = 0.0f;

    for (int beta = 1; beta < 256; beta++) {
        int f[256];

        for (int x = 0; x < 256; x++) {
            f[x] = __builtin_parity(beta & Y[x]) ? -1 : 1;
        }

        fwt_int_256(f);

        for (int alpha = 1; alpha < 256; alpha++) {
            float delta_alpha_beta = -(float)f[alpha] * norm;
            float abs_delta = delta_alpha_beta < 0 ? -delta_alpha_beta : delta_alpha_beta;
            if (abs_delta > max_abs_delta) max_abs_delta = abs_delta;
        }
    }
    return max_abs_delta;
}


int main(void) {
    uint8_t T = 9;
    uint8_t r[10] = {2, 0, 3, 3, 2, 1, 3, 3, 2};
    uint8_t S[10] = {1, 2, 0, 3, 0, 0, 0, 0, 0};

    uint8_t sbox[256];
    build_sbox(sbox, r, S, T);

    if (!is_permutation(sbox)) {
        printf("Not a permutation for T=%u\n", T);
        return 0;
    }

    float p_pi      = compute_p_pi(sbox);
    int   lambda_pi = compute_lambda(sbox);
    float delta_pi  = compute_delta(sbox);

    printf("T = %u\n", T);
    printf("r = [");
    for (int i = 0; i < T; ++i) printf("%u%s", r[i], (i+1<T)?", ":"");
    printf("]\nS = [");
    for (int i = 0; i < T; ++i) printf("%u%s", S[i], (i+1<T)?", ":"");
    printf("]\n");

    printf("p_pi      = %.7f\n", p_pi);
    printf("lambda_pi = %d\n",   lambda_pi);
    printf("delta_pi  = %.7f\n", delta_pi);

    return 0;
}
