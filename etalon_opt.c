#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <immintrin.h>
#include <omp.h>
#include <stdalign.h>

#define TOP_SIZE 5
#define MAX_T 10

typedef struct {
    float prob;
    int lambda_pi;
    float linear_bias; 
    uint8_t r[MAX_T];
    uint8_t S[MAX_T];
} Combination;

static uint8_t cls_table[16][4];
static uint8_t A_single_cache[16][16][4][4];
static bool tables_initialized = false;

void init_tables() {
    if (tables_initialized) return;
    

    for (uint8_t val = 0; val < 16; ++val) {
        for (uint8_t shift = 0; shift < 4; ++shift) {
            cls_table[val][shift] = ((val << shift) | ((val << shift) >> 4)) & 0x0F;
        }
    }
    

    for (uint8_t X1 = 0; X1 < 16; ++X1) {
        for (uint8_t X2 = 0; X2 < 16; ++X2) {
            for (uint8_t r = 0; r < 4; ++r) {
                for (uint8_t S = 0; S < 4; ++S) {
                    uint8_t Z = (X1 + cls_table[X2][r]) & 0x0F;
                    uint8_t Y2 = (X2 ^ cls_table[Z][S]) & 0x0F;
                    A_single_cache[X1][X2][r][S] = (Z << 4) | Y2;
                }
            }
        }
    }
    tables_initialized = true;
}

static inline uint8_t A_single_opt(const uint8_t X, const uint8_t r, const uint8_t S) {
    return A_single_cache[(X >> 4) & 0x0F][X & 0x0F][r][S];
}

static inline int is_permutation_opt(const uint8_t *sbox) {
    uint64_t masks[4] = {0};
    for (int i = 0; i < 256; ++i) {
        uint8_t val = sbox[i];
        uint64_t bit = 1ULL << (val & 0x3F);
        int idx = val >> 6;
        if (masks[idx] & bit) return 0;
        masks[idx] |= bit;
    }
    return 1;
}

static inline uint8_t A(const uint8_t X, const uint8_t *r, const uint8_t *S, const uint8_t T) {
    uint8_t Y = A_single_opt(X, r[0], S[0]);
    for (int i = 1; i < T; ++i) 
        Y = A_single_opt(Y, r[i], S[i]);
    return Y;
}

static void compute_sbox(uint8_t *sbox, const uint8_t T, const uint8_t *r, const uint8_t *S) {
    #pragma omp simd
    for (uint16_t x = 0; x < 256; ++x) 
        sbox[x] = A(x, r, S, T);
}

static float compute_differential_prob_opt(const uint8_t *sbox) {
    alignas(32) uint16_t P[256][256] = {0};
    float max_prob = 0.0f;

    #pragma omp parallel for reduction(max:max_prob)
    for (int a = 1; a < 256; ++a) {
        uint16_t local_P[256] = {0};
        for (int x = 0; x < 256; ++x) {
            int x_xor_a = x ^ a;
            local_P[sbox[x] ^ sbox[x_xor_a]]++;
        }
        
        for (int b = 0; b < 256; ++b) {
            float prob = (float)local_P[b] / 256.0f;
            if (prob > max_prob) max_prob = prob;
        }
    }
    return max_prob;
}


static uint8_t parity(uint8_t x) {
    x ^= x >> 4;
    x ^= x >> 2;
    x ^= x >> 1;
    return x & 1;
}

static int count_bits(uint8_t x) {
    x = (x & 0x55) + ((x >> 1) & 0x55);
    x = (x & 0x33) + ((x >> 2) & 0x33);
    return (x & 0x0F) + (x >> 4);
}

uint8_t compute_lambda_pi_opt(const uint8_t sbox[256]) {
    uint8_t min_degree = 8;

    #pragma omp parallel for reduction(min:min_degree)
    for (uint16_t alpha = 1; alpha < 256; ++alpha) {
        uint8_t anf[256] = {0};
        for (int x = 0; x < 256; ++x) 
            anf[x] = parity(alpha & sbox[x]);

        for (int i = 0; i < 8; ++i) 
            for (int j = 0; j < 256; ++j) 
                if (j & (1 << i)) 
                    anf[j] ^= anf[j ^ (1 << i)];

        uint8_t current_degree = 0;
        for (int j = 0; j < 256; ++j) {
            if (anf[j]) {
                int bits = count_bits(j);
                current_degree = bits > current_degree ? bits : current_degree;
            }
        }

        if (current_degree < min_degree) 
            min_degree = current_degree;
    }
    return min_degree;
}


static float compute_delta_opt(const uint8_t sbox[256]) {
    alignas(32) int f[256];
    float max_delta = 0.0f;

    #pragma omp parallel for reduction(max:max_delta)
    for (int beta = 1; beta < 256; beta++) {
        for (int x = 0; x < 256; x++) {
            f[x] = (parity(beta & sbox[x]) == 0) ? 1 : -1;
        }

        int len = 128;
        while (len > 0) {
            for (int start = 0; start < 256; start += 2 * len) {
                for (int i = start; i < start + len; i++) {
                    int a = f[i];
                    int b = f[i + len];
                    f[i] = a + b;
                    f[i + len] = a - b;
                }
            }
            len >>= 1;
        }

        for (int alpha = 1; alpha < 256; alpha++) {
            float bias = fabsf((float)f[alpha]) / 256.0f;
            if (bias > max_delta) {
                max_delta = bias;
            }
        }
    }
    return max_delta;
}


static int is_pareto_dominant(const Combination *a, const Combination *b) {
    return (a->prob <= b->prob) && 
           (a->linear_bias <= b->linear_bias) && 
           (a->lambda_pi >= b->lambda_pi) &&
           ((a->prob < b->prob) || 
            (a->linear_bias < b->linear_bias) || 
            (a->lambda_pi > b->lambda_pi));
}

void update_pareto_front(Combination *front, int *count, Combination candidate) {
    bool dominated = false;

    #pragma omp critical
    {

        for (int i = 0; i < *count; ++i) {
            if (is_pareto_dominant(&front[i], &candidate)) {
                dominated = true;
                break;
            }
        }

        if (!dominated) {
            Combination new_front[TOP_SIZE];
            int new_count = 0;

            for (int i = 0; i < *count; ++i) {
                if (!is_pareto_dominant(&candidate, &front[i])) {
                    new_front[new_count++] = front[i];
                }
            }

            if (new_count < TOP_SIZE) {
                new_front[new_count++] = candidate;
            }

            memcpy(front, new_front, new_count * sizeof(Combination));
            *count = new_count;
        }
    }
}

void process_all_combinations(uint8_t T) {
    init_tables();
    char filename[26];
    snprintf(filename, sizeof(filename), "Combinations_T%d.txt", T);
    FILE *file = fopen(filename, "w");
    if (!file) { perror("Error"); return; }

    const uint64_t total_combinations = 1ULL << (4 * T);
    Combination pareto_front[TOP_SIZE] = {{0}};
    int pareto_count = 0;

    time_t start_time = time(NULL);
    fprintf(file, "Start time: %s", ctime(&start_time));
    printf("T=%d (%llu comb) | Threads: %d\n", T, total_combinations, omp_get_max_threads());

    #pragma omp parallel for schedule(dynamic)
    for (uint64_t combo = 0; combo < total_combinations; ++combo) {
        uint8_t r[MAX_T], S[MAX_T];
        uint8_t sbox[256];
        uint64_t tmp = combo;
        

        for (int i = 0; i < T; ++i) {
            r[i] = tmp & 0x03;
            tmp >>= 2;
        }
        for (int i = 0; i < T; ++i) {
            S[i] = tmp & 0x03;
            tmp >>= 2;
        }

        compute_sbox(sbox, T, r, S);
        if (!is_permutation_opt(sbox)) continue;

        Combination current;
        current.prob = compute_differential_prob_opt(sbox);
        current.lambda_pi = compute_lambda_pi_opt(sbox);
        current.linear_bias = compute_delta_opt(sbox); 
        memcpy(current.r, r, T);
        memcpy(current.S, S, T);
        
        update_pareto_front(pareto_front, &pareto_count, current);


        if (combo % (1ULL << 20) == 0) {
            #pragma omp critical
            printf("\rProgress: %.2f%%", (double)combo / total_combinations * 100);
        }
    }

    // Sort Pareto front
    for (int i = 0; i < pareto_count; ++i) {
        for (int j = i+1; j < pareto_count; ++j) {
            if ((pareto_front[i].prob > pareto_front[j].prob) || 
                (pareto_front[i].prob == pareto_front[j].prob && 
                 pareto_front[i].lambda_pi < pareto_front[j].lambda_pi)) {
                Combination tmp = pareto_front[i];
                pareto_front[i] = pareto_front[j];
                pareto_front[j] = tmp;
            }
        }
    }


    fprintf(file, "=== Pareto optimal combinations (T=%d) ===\n", T);
    for (int i = 0; i < pareto_count; ++i) {
        fprintf(file, "r = [");
        for (int j = 0; j < T; ++j) 
            fprintf(file, "%d ", pareto_front[i].r[j]);
        fprintf(file, "], S = [");
        for (int j = 0; j < T; ++j) 
            fprintf(file, "%d ", pareto_front[i].S[j]);
        fprintf(file, "], p_pi = %.7f, lambda_π = %d, delta_π = %.7f\n", 
                pareto_front[i].prob, 
                pareto_front[i].lambda_pi,
                pareto_front[i].linear_bias); 
    }

    time_t end_time = time(NULL);
    fprintf(file, "\nTotal time: %.2f seconds\n", difftime(end_time, start_time));
    fclose(file);
    printf("\nResults saved to %s\n", filename);
}

int main() {
    uint8_t T;
    printf("Enter T: ");
    if (scanf("%hhu", &T) != 1 || T < 1 || T > 10) {
        printf("Invalid T\n");
        return 1;
    }
    process_all_combinations(T);
    return 0;
}