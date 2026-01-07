#define NOMINMAX

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <fcntl.h>
#endif


#define TOP_SIZE 20
#define MAX_T    20


#define METRIC_PPI_ONLY    0
#define METRIC_DELTA_ONLY  1
#define METRIC_ALL         2
#define METRIC_HIST        3
#define METRIC_ALL_HIST    4

struct Combination {
    float   p_pi;
    int     lambda_pi;
    float   delta_pi;
    uint8_t r[MAX_T];
    uint8_t S[MAX_T];
};

struct DeviceOut {
    float   *p_pi;
    int     *lambda_pi;
    float   *delta_pi;
    uint8_t *rS;       // [iters][2*MAX_T]
    uint8_t *is_valid; // [iters]
};

// Для histogram: ключ = (ddt_max, lambda, lat_max)
struct HistKey {
    uint16_t ddt_max;  // raw DDT max
    int      lambda;   // algebraic degree
    uint16_t lat_max;  // raw LAT max

    bool operator<(const HistKey &o) const {
        if (ddt_max != o.ddt_max) return ddt_max < o.ddt_max;
        if (lambda  != o.lambda)  return lambda  < o.lambda;
        return lat_max < o.lat_max;
    }
};


__constant__ uint8_t d_A_single_cache[16][16][4][4];


static uint8_t h_A_single_cache[16][16][4][4];

static void init_tables_host() {
    static bool inited = false;
    if (inited) return;

    uint8_t cls_table[16][4];
    for (uint8_t val = 0; val < 16; ++val)
        for (uint8_t shift = 0; shift < 4; ++shift)
            cls_table[val][shift] = (uint8_t)(((val << shift) | ((val << shift) >> 4)) & 0x0F);

    for (uint8_t X1 = 0; X1 < 16; ++X1)
        for (uint8_t X2 = 0; X2 < 16; ++X2)
            for (uint8_t r = 0; r < 4; ++r)
                for (uint8_t S = 0; S < 4; ++S) {
                    uint8_t Z  = (uint8_t)((X1 + cls_table[X2][r]) & 0x0F);
                    uint8_t Y2 = (uint8_t)((X2 ^ cls_table[Z][S]) & 0x0F);
                    h_A_single_cache[X1][X2][r][S] = (uint8_t)((Z << 4) | Y2);
                }

    inited = true;
}

static void upload_tables_to_device() {
    init_tables_host();
    cudaMemcpyToSymbol(d_A_single_cache, h_A_single_cache, sizeof(h_A_single_cache));
}


__device__ __forceinline__ uint8_t A_single_opt_dev(uint8_t X, uint8_t r, uint8_t S) {
    return d_A_single_cache[(X >> 4) & 0x0F][X & 0x0F][r][S];
}

__device__ __forceinline__ uint8_t A_dev(uint8_t X, const uint8_t *r, const uint8_t *S, uint8_t T) {
    uint8_t Y = A_single_opt_dev(X, r[0], S[0]);
    #pragma unroll
    for (int i = 1; i < MAX_T; ++i) {
        if (i >= T) break;
        Y = A_single_opt_dev(Y, r[i], S[i]);
    }
    return Y;
}

__device__ __forceinline__ int is_permutation_dev(const uint8_t sbox[256]) {
    uint64_t masks[4] = {0,0,0,0};
    #pragma unroll 1
    for (int i = 0; i < 256; ++i) {
        uint8_t val = sbox[i];
        uint64_t bit = 1ULL << (val & 0x3F);
        int idx = val >> 6;
        if (masks[idx] & bit) return 0;
        masks[idx] |= bit;
    }
    return 1;
}

__device__ __forceinline__ uint8_t parity8(uint8_t x) {
    x ^= x >> 4; x ^= x >> 2; x ^= x >> 1;
    return x & 1;
}

__device__ __forceinline__ int popcnt8(uint8_t x) {
    x = (x & 0x55) + ((x >> 1) & 0x55);
    x = (x & 0x33) + ((x >> 2) & 0x33);
    return (x & 0x0F) + (x >> 4);
}


__device__ uint16_t compute_ddt_max_dev(const uint8_t sbox[256]) {
    uint16_t global_max = 0, row[256];
    for (int a = 1; a < 256; ++a) {
        #pragma unroll 1
        for (int i = 0; i < 256; ++i) row[i] = 0;
        #pragma unroll 1
        for (int x = 0; x < 256; ++x) {
            uint8_t b = sbox[x] ^ sbox[x ^ a];
            row[b]++;
        }
        #pragma unroll 1
        for (int b = 0; b < 256; ++b)
            if (row[b] > global_max) global_max = row[b];
    }
    return global_max;
}

__device__ int16_t compute_lat_max_abs_dev(const uint8_t sbox[256]) {
    int16_t max_abs = 0, f[256];
    for (int beta = 1; beta < 256; ++beta) {
        for (int x = 0; x < 256; ++x)
            f[x] = (parity8((uint8_t)(beta & sbox[x])) == 0) ? 1 : -1;
        for (int len = 1; len < 256; len <<= 1)
            for (int i = 0; i < 256; i += (len << 1))
                for (int j = 0; j < len; ++j) {
                    int16_t u = f[i+j], v = f[i+j+len];
                    f[i+j]     = (int16_t)(u + v);
                    f[i+j+len] = (int16_t)(u - v);
                }
        for (int alpha = 1; alpha < 256; ++alpha) {
            int16_t v = f[alpha]; if (v < 0) v = -v;
            if (v > max_abs) max_abs = v;
        }
    }
    return max_abs;
}

__device__ int compute_lambda_min_degree_dev(const uint8_t sbox[256]) {
    uint8_t coord[8][256];
    for (int x = 0; x < 256; ++x) {
        uint8_t y = sbox[x];
        #pragma unroll
        for (int i = 0; i < 8; ++i) coord[i][x] = (y >> i) & 1;
    }
    auto mobius = [&] (uint8_t f[256]) {
        for (int i = 0; i < 8; ++i)
            for (int m = 0; m < 256; ++m)
                if (m & (1 << i)) f[m] ^= f[m ^ (1 << i)];
    };
    for (int i = 0; i < 8; ++i) mobius(coord[i]);

    uint64_t bits[8][9][4] = {};
    for (int i = 0; i < 8; ++i)
        for (int m = 0; m < 256; ++m) if (coord[i][m]) {
            int d = popcnt8((uint8_t)m);
            bits[i][d][m >> 6] |= 1ULL << (m & 63);
        }

    int min_deg = 8;
    for (int a = 1; a < 256; ++a) {
        uint64_t acc[9][4] = {};
        for (int i = 0; i < 8; ++i) if (a & (1 << i))
            for (int d = 0; d <= 8; ++d)
                for (int w = 0; w < 4; ++w)
                    acc[d][w] ^= bits[i][d][w];

        int cur_deg = 0;
        for (int d = 8; d >= 0; --d) {
            uint64_t orr = acc[d][0] | acc[d][1] | acc[d][2] | acc[d][3];
            if (orr) { cur_deg = d; break; }
        }
        if (cur_deg < min_deg) min_deg = cur_deg;
    }
    return min_deg;
}



__global__ void enumerateKernel(
    uint8_t T, uint64_t start_index, uint64_t count, int metric_mode, DeviceOut out)
{
    uint64_t gid  = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t gdim = gridDim.x * (uint64_t)blockDim.x;

    for (uint64_t off = gid; off < count; off += gdim) {
        uint64_t idx = start_index + off;
        uint8_t r[MAX_T], S[MAX_T], sbox[256];
        uint64_t tmp = idx;
        for (int i = 0; i < T; ++i) { r[i] = tmp & 0x03; tmp >>= 2; }
        for (int i = 0; i < T; ++i) { S[i] = tmp & 0x03; tmp >>= 2; }
        for (int x = 0; x < 256; ++x)
            sbox[x] = A_dev((uint8_t)x, r, S, T);

        if (!is_permutation_dev(sbox)) {
            out.is_valid[off] = 0;
            continue;
        }

        float p_pi_f   = 1e30f;
        float delta_f  = 1e30f;
        int   lambdaMin= -1;

        if (metric_mode == METRIC_PPI_ONLY) {
            uint16_t ddt = compute_ddt_max_dev(sbox);
            p_pi_f = ddt / 256.0f;
        } else if (metric_mode == METRIC_DELTA_ONLY) {
            int16_t lat = compute_lat_max_abs_dev(sbox);
            delta_f = lat / 256.0f;
        } else {
            uint16_t ddt = compute_ddt_max_dev(sbox);
            int16_t  lat = compute_lat_max_abs_dev(sbox);
            lambdaMin   = compute_lambda_min_degree_dev(sbox);
            p_pi_f      = ddt / 256.0f;
            delta_f     = lat / 256.0f;
        }

        out.p_pi[off]      = p_pi_f;
        out.delta_pi[off]  = delta_f;
        out.lambda_pi[off] = lambdaMin;

        uint8_t *dst = out.rS + off * (2 * MAX_T);
        for (int i = 0; i < MAX_T; ++i)         dst[i]         = (i < T ? r[i] : 0);
        for (int i = 0; i < MAX_T; ++i)         dst[MAX_T + i] = (i < T ? S[i] : 0);

        out.is_valid[off] = 1;
    }
}


static inline bool pareto_dominates(const Combination &a, const Combination &b) {
    bool better_or_equal = (a.p_pi <= b.p_pi) &&
                           (a.delta_pi <= b.delta_pi) &&
                           (a.lambda_pi >= b.lambda_pi);
    bool strictly_better = (a.p_pi <  b.p_pi) ||
                           (a.delta_pi <  b.delta_pi) ||
                           (a.lambda_pi >  b.lambda_pi);
    return better_or_equal && strictly_better;
}

static void update_pareto_front(std::vector<Combination> &front,
                                const Combination &cand)
{
    for (auto &f : front)
        if (pareto_dominates(f, cand))
            return;
    std::vector<Combination> nf;
    nf.reserve(TOP_SIZE);
    for (auto &f : front)
        if (!pareto_dominates(cand, f))
            nf.push_back(f);
    if ((int)nf.size() < TOP_SIZE)
        nf.push_back(cand);
    front.swap(nf);
}

static void update_top_p(std::vector<Combination> &top_p,
                         const Combination &cand)
{
    auto it = std::lower_bound(
        top_p.begin(), top_p.end(), cand,
        [] (auto &a, auto &b) { return a.p_pi < b.p_pi; }
    );
    if (it != top_p.end() || top_p.size() < TOP_SIZE) {
        top_p.insert(it, cand);
        if (top_p.size() > TOP_SIZE)
            top_p.pop_back();
    }
}

static void update_top_delta(std::vector<Combination> &top_d,
                             const Combination &cand)
{
    auto it = std::lower_bound(
        top_d.begin(), top_d.end(), cand,
        [] (auto &a, auto &b) { return a.delta_pi < b.delta_pi; }
    );
    if (it != top_d.end() || top_d.size() < TOP_SIZE) {
        top_d.insert(it, cand);
        if (top_d.size() > TOP_SIZE)
            top_d.pop_back();
    }
}


static void write_results(
    const std::string &filename,
    const std::vector<Combination> &pareto,
    const std::vector<Combination> &top_p,
    const std::vector<Combination> &top_d,
    uint8_t T,
    time_t start, time_t end,
    unsigned long long checked)
{
    std::ofstream file(filename);
    if (!file) { std::perror("Error opening output file"); return; }

    auto fmt_float = [] (float v) -> std::string {
        if (!(v < 1e29f)) return "NA";
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(7) << v;
        return ss.str();
    };


    file << "=== Pareto optimal combinations (T=" << int(T) << ") ===\n";
    if (pareto.empty()) {
        file << "(none found)\n";
    } else {
        for (auto &c : pareto) {
            file << "r=[";
            for (int j = 0; j < T; ++j) file << int(c.r[j]) << " ";
            file << "], S=[";
            for (int j = 0; j < T; ++j) file << int(c.S[j]) << " ";
            file << "], p_pi=" << fmt_float(c.p_pi)
                 << ", lambda_pi=" 
                 << (c.lambda_pi >= 0 ? std::to_string(c.lambda_pi) : "NA")
                 << ", delta_pi=" << fmt_float(c.delta_pi) << "\n";
        }
    }


    file << "\n=== Top " << TOP_SIZE << " minimal p_pi (with r/S) ===\n";
    if (top_p.empty()) {
        file << "(none found)\n";
    } else {
        for (auto &c : top_p) {
            file << "r=[";
            for (int j = 0; j < T; ++j) file << int(c.r[j]) << " ";
            file << "], S=[";
            for (int j = 0; j < T; ++j) file << int(c.S[j]) << " ";
            file << "], p_pi=" << fmt_float(c.p_pi)
                 << ", lambda_pi=" 
                 << (c.lambda_pi >= 0 ? std::to_string(c.lambda_pi) : "NA")
                 << ", delta_pi=" << fmt_float(c.delta_pi) << "\n";
        }
    }


    file << "\n=== Top " << TOP_SIZE << " minimal delta_pi (with r/S) ===\n";
    if (top_d.empty()) {
        file << "(none found)\n";
    } else {
        for (auto &c : top_d) {
            file << "r=[";
            for (int j = 0; j < T; ++j) file << int(c.r[j]) << " ";
            file << "], S=[";
            for (int j = 0; j < T; ++j) file << int(c.S[j]) << " ";
            file << "], p_pi=" << fmt_float(c.p_pi)
                 << ", lambda_pi=" 
                 << (c.lambda_pi >= 0 ? std::to_string(c.lambda_pi) : "NA")
                 << ", delta_pi=" << fmt_float(c.delta_pi) << "\n";
        }
    }

    file << "\n---\n";
    file << "Checked combinations: " << checked << "\n";
    file << "Start time: " << ctime(&start);
    file << "End time:   " << ctime(&end);
    file << "Elapsed:    " << std::fixed << std::setprecision(2)
         << difftime(end, start) << " sec\n";
    std::cout << "Results saved to " << filename << "\n";
}

static void write_histogram(
    const std::string &filename,
    const std::map<HistKey,uint64_t> &hist,
    uint64_t total,
    uint8_t T,
    time_t start, time_t end)
{
    std::ofstream file(filename);
    if (!file) { std::perror("Error opening histogram file"); return; }

    file << "=== Histogram of combinations (T=" << int(T) << ") ===\n";
    file << "p_pi\tlambda_pi\tdelta_pi\tcount\tpercent\n";
    for (auto &kv : hist) {
        float   p   = kv.first.ddt_max / 256.0f;
        int     lam = kv.first.lambda;
        float   d   = kv.first.lat_max / 256.0f;
        uint64_t cnt= kv.second;
        double  pct = 100.0 * cnt / (double)total;
        file << std::fixed << std::setprecision(7)
             << p   << "\t"
             << lam << "\t"
             << d   << "\t"
             << cnt << "\t"
             << std::fixed << std::setprecision(4)
             << pct << "%\n";
    }
    file << "---\n";
    file << "Total valid combinations: " << total << "\n";
    file << "Start time: " << ctime(&start);
    file << "End time:   " << ctime(&end);
    file << "Elapsed:    " << std::fixed << std::setprecision(2)
         << difftime(end, start) << " sec\n";
    std::cout << "Histogram saved to " << filename << "\n";
}


static void run_full_enumeration(uint8_t T, int metric_mode) {
    upload_tables_to_device();

    
    if (4ULL * T >= 64) {
        return;
    }

    const uint64_t total = 1ULL << (4 * T);
    const uint64_t chunk = 1ULL << 22;

    dim3 block(128);
    dim3 grid(static_cast<unsigned int>(
        std::min<uint64_t>((chunk + block.x - 1) / block.x, 65535ULL)
    ));

    std::vector<uint8_t> h_valid(chunk);
    std::vector<float>   h_p(chunk), h_d(chunk);
    std::vector<int>     h_l(chunk);
    std::vector<uint8_t> h_rS(chunk * 2 * MAX_T);

    DeviceOut d{};
    cudaMalloc(&d.p_pi,      chunk * sizeof(float));
    cudaMalloc(&d.lambda_pi, chunk * sizeof(int));
    cudaMalloc(&d.delta_pi,  chunk * sizeof(float));
    cudaMalloc(&d.rS,        chunk * 2 * MAX_T * sizeof(uint8_t));
    cudaMalloc(&d.is_valid,  chunk * sizeof(uint8_t));

    std::vector<Combination> pareto, top_p, top_d;
    std::map<HistKey,uint64_t> hist;
    uint64_t total_checked = 0;
    time_t start = time(NULL);

    for (uint64_t start_idx = 0; start_idx < total; start_idx += chunk) {
        uint64_t count = std::min<uint64_t>(chunk, total - start_idx);
        
        cudaMemset(d.is_valid, 0, count * sizeof(uint8_t));

        enumerateKernel<<<grid, block>>>(T, start_idx, count, metric_mode, d);
        cudaDeviceSynchronize();

        
        cudaMemcpy(h_valid.data(), d.is_valid,   count * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_p.data(),     d.p_pi,       count * sizeof(float),   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_l.data(),     d.lambda_pi,  count * sizeof(int),     cudaMemcpyDeviceToHost);
        cudaMemcpy(h_d.data(),     d.delta_pi,   count * sizeof(float),   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_rS.data(),    d.rS,         count * 2 * MAX_T * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        for (uint64_t i = 0; i < count; ++i) {
            if (!h_valid[i]) continue;
            ++total_checked;
            Combination c{};
            c.p_pi      = h_p[i];
            c.lambda_pi = h_l[i];
            c.delta_pi  = h_d[i];
            for (int j = 0; j < T; ++j) {
                c.r[j] = h_rS[i*2*MAX_T + j];
                c.S[j] = h_rS[i*2*MAX_T + MAX_T + j];
            }

            bool do_pareto = (metric_mode == METRIC_ALL) || (metric_mode == METRIC_ALL_HIST) ||
                             (metric_mode == METRIC_PPI_ONLY) || (metric_mode == METRIC_DELTA_ONLY);
            bool do_top_p  = (metric_mode != METRIC_DELTA_ONLY);
            bool do_top_d  = (metric_mode != METRIC_PPI_ONLY);

            if (metric_mode != METRIC_HIST) {
                if (do_pareto) update_pareto_front(pareto, c);
                if (do_top_p && metric_mode != METRIC_DELTA_ONLY) update_top_p(top_p, c);
                if (do_top_d && metric_mode != METRIC_PPI_ONLY) update_top_delta(top_d, c);
            }

            
            if (metric_mode == METRIC_HIST || metric_mode == METRIC_ALL_HIST) {
                
                uint16_t p_raw = 0, d_raw = 0;
                if (c.p_pi < 1e29f) p_raw = static_cast<uint16_t>(std::lround(c.p_pi * 256.0f));
                if (c.delta_pi < 1e29f) d_raw = static_cast<uint16_t>(std::lround(c.delta_pi * 256.0f));
                HistKey key{ p_raw, c.lambda_pi, d_raw };
                hist[key]++;
            }
        }

        if (((start_idx / chunk) % 50) == 0) {
            double pct = 100.0 * (double)(start_idx + count) / (double)total;
            std::cerr << "\rProgress: "
                      << std::fixed << std::setprecision(2)
                      << pct << "%   " << std::flush;
        }
    }
    std::cerr << "\n";

    time_t end = time(NULL);
    cudaFree(d.p_pi);
    cudaFree(d.lambda_pi);
    cudaFree(d.delta_pi);
    cudaFree(d.rS);
    cudaFree(d.is_valid);

    char fname[128];
    if (metric_mode != METRIC_HIST && metric_mode != METRIC_ALL_HIST) {
        std::snprintf(fname, sizeof(fname), "combinations_T%u_cuda.txt", (unsigned)T);
        write_results(fname, pareto, top_p, top_d, T, start, end, total_checked);
    } else if (metric_mode == METRIC_HIST) {
        std::snprintf(fname, sizeof(fname), "histogram_T%u_cuda.txt", (unsigned)T);
        write_histogram(fname, hist, total_checked, T, start, end);
    } else { 
        std::snprintf(fname, sizeof(fname), "combinations_T%u_cuda.txt", (unsigned)T);
        write_results(fname, pareto, top_p, top_d, T, start, end, total_checked);
        std::snprintf(fname, sizeof(fname), "histogram_T%u_cuda.txt", (unsigned)T);
        write_histogram(fname, hist, total_checked, T, start, end);
    }
}


int main() {
    unsigned int T = 1;
    std::cout << "Enter T: ";
    if (!(std::cin >> T)) return 1;
    if (T < 1 || T > 20) { return 1; }

    

    int metric_mode = METRIC_ALL;
    std::cout << "Select metric mode (0 - p_pi only, 1 - delta_pi only, 2 - all, 3 - histogram, 4 - all+hist): ";
    if (!(std::cin >> metric_mode)) return 1;
    if (metric_mode < METRIC_PPI_ONLY || metric_mode > METRIC_ALL_HIST) {
        std::cout << "Invalid metric mode\n";
        return 1;
    }

    run_full_enumeration((uint8_t)T, metric_mode);

    return 0;
}
