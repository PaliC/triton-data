import torch
import json
import heapq
import triton
import triton.language as tl
from triton.runtime import driver

def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()

def get_configs_io_bound():
    configs = []
    for num_stages in [4, 5]:
        for block_m in [16,32,64,128]:
            for block_k in [32,64]:
                for block_n in [64,128,256]:
                    for group_m in [4,8,16]:
                        num_warps = 2 if block_n <= 64 else 4
                        configs.append(
                            triton.Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': block_k, 'GROUP_SIZE_M':group_m},
                                      num_stages=num_stages, num_warps=num_warps))
    return configs

configs = get_configs_io_bound()
device = torch.cuda.current_device()
capability = torch.cuda.get_device_capability()
a = torch.randn((512,512), device='cuda', dtype=torch.float16)
dtsize = a.element_size()
dtype = a.dtype
pruned_configs = []
for config in configs:
    kw = config.kwargs
    BLOCK_M, BLOCK_N, BLOCK_K, num_stages = \
        kw['BLOCK_SIZE_M'], kw['BLOCK_SIZE_N'], kw['BLOCK_SIZE_K'], config.num_stages

    max_shared_memory = driver.utils.get_device_properties(device)["max_shared_mem"]
    required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
    if required_shared_memory <= max_shared_memory:
        pruned_configs.append(config)

configs = pruned_configs
configs_map = {}
for config in configs:
    kw = config.kwargs
    BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages = \
        kw['BLOCK_SIZE_M'], kw['BLOCK_SIZE_N'], kw['BLOCK_SIZE_K'],kw['GROUP_SIZE_M'], config.num_warps, config.num_stages

    key = (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps)
    if key in configs_map:
        configs_map[key].append((config, num_stages))
    else:
        configs_map[key] = [(config, num_stages)]

pruned_configs = []
for k, v in configs_map.items():
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps = k
    if capability[0] >= 8:
        # compute cycles (only works for ampere GPUs)
        mmas = BLOCK_M * BLOCK_N * BLOCK_K / (16 * 8 * 16)
        mma_cycles = mmas / min(4, num_warps) * 8

        ldgsts_latency = 300  # Does this matter?
        optimal_num_stages = ldgsts_latency / mma_cycles

        # nearest stages, prefer large #stages
        nearest = heapq.nsmallest(2, v, key=lambda x: 10 + abs(x[1] - optimal_num_stages)
                                    if (x[1] - optimal_num_stages) < 0 else x[1] - optimal_num_stages)

        for n in nearest:
            pruned_configs.append(n[0])
    else:  # Volta & Turing only supports num_stages <= 2
        random_config = v[0][0]
        random_config.num_stages = 2
        pruned_configs.append(random_config)

@triton.autotune(
    configs= pruned_configs,
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)
def matmul(a, b, activation=""):

    assert a.shape[1] == b.shape[0], "维度符合矩阵相乘要求"
    assert a.is_contiguous(), "矩阵A必须是连续的"
    assert b.is_contiguous(), "矩阵B必须是连续的"
    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
 
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation
    )
    return c

def extract_numbers(config_str,numbers):
    for part in config_str.split(', '):
        if ':' in part:
            key, value = part.split(': ')
            try:
                number = int(value)
                numbers.append(number)
            except ValueError:
                pass 

numbers = []
m_low=4;m_up=128;m_step=4
n_low=8192;n_up=8192;n_step=128
k_low=12288;k_up=12288;k_step=128
for m in range(m_low,m_up+m_step,m_step):
    for n in range(n_low,n_up+n_step,n_step):
            for k in range(k_low,k_up+k_step,k_step):
                a1 = torch.randn((m,k), device='cuda', dtype=torch.float16)
                b1 = torch.randn((k,n), device='cuda', dtype=torch.float16)
                triton_output = matmul(a1, b1)
                print(f"Best config of meta parameters for M={m},N={n},K={k} are shown below: ",)
                print(matmul_kernel.best_config)
                config_str=str(matmul_kernel.best_config)
                numbers.append(m)
                numbers.append(n)
                numbers.append(k)
                extract_numbers(config_str,numbers)

m_low=8;m_up=13;m_step=1
n_low=8192;n_up=8192;n_step=128
k_low=12288;k_up=12288;k_step=128
for m in range(m_low,m_up+m_step,m_step):
    for n in range(n_low,n_up+n_step,n_step):
            for k in range(k_low,k_up+k_step,k_step):
                a1 = torch.randn((2**m,k), device='cuda', dtype=torch.float16)
                b1 = torch.randn((k,n), device='cuda', dtype=torch.float16)
                triton_output = matmul(a1, b1)
                print(f"Best config of meta parameters for M={2**m},N={n},K={k} are shown below: ",)
                print(matmul_kernel.best_config)
                config_str=str(matmul_kernel.best_config)
                numbers.append(2**m)
                numbers.append(n)
                numbers.append(k)
                extract_numbers(config_str,numbers)

print(numbers)
labels = ['M','N','K','BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K',' GROUP_SIZE_M','num_warps','num_stages']
grouped_data = []
for i in range(0, len(numbers), 9):
        group = {label: numbers[i+j] for j, label in enumerate(labels)}
        grouped_data.append(group)
with open("meta_config.json", "w") as f:
    json.dump(grouped_data, f, indent=2)
