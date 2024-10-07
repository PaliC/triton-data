import torch
import triton
import triton.language as tl

MIN_BLOCK_SIZE = 16
ALLOW_TF32 = False


@triton.jit
def _depth_lin(near, far, n, step):
    frac_step = step / (n - 1)
    return (far - near) * frac_step + near


@triton.jit
def _depth_inv_sphere(far, disparity_at_inf, n, step):
    frac_step = (step + 1) / n
    n_disp = (disparity_at_inf - 1) * frac_step + 1
    return far * (1 / n_disp)


@triton.jit
def _contract_pi(
    x,
    y,
    z,
    a,
):
    # MERF contract_pi function
    # def contract_pi(x):
    #     n = x.abs().max(dim=-1).values[..., None]
    #     x_contract = torch.where(
    #         n <= 1.0,
    #         x,
    #         torch.where(
    #             (x.abs()-n).abs() <= 1e-7,
    #             (2 - 1/x.abs()) * (x / x.abs()),
    #             x / n,
    #         )
    #     )
    #     return x_contract
    n = tl.maximum(tl.maximum(tl.abs(x), tl.abs(y)), tl.abs(z))
    x_c = _contract_pi_one(x, n, a)
    y_c = _contract_pi_one(y, n, a)
    z_c = _contract_pi_one(z, n, a)
    return x_c, y_c, z_c


@triton.jit
def _contract_pi_one(x, n, a):
    x_c = tl.where(
        tl.abs(x) <= 1.0,
        x * a,
        ((2 - a) * (1 - 1 / tl.abs(x)) + a) * (x / tl.abs(x)),
    )
    # important: we map the contracted coords from [-2, 2] to [-1, 1]!
    x_c = x_c * 0.5
    return x_c


@triton.jit
def _sample_2d(
    image,
    w,
    batch_index,
    ix,
    iy,
    IH: tl.constexpr,
    IW: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    channel_bcast = tl.full((1, C), 1.0, dtype=tl.float32)
    Coffs = tl.arange(0, C)
    ix_ = tl.minimum(tl.maximum(ix, 0.0), IW - 1).to(tl.int32)
    iy_ = tl.minimum(tl.maximum(iy, 0.0), IH - 1).to(tl.int32)
    val = tl.view(
        tl.load(
            (image + batch_index * IW * IH * C + iy_ * IW * C + ix_ * C)[:, None]
            + Coffs[None, :]
        ),
        (BLOCK_SIZE, C),
    )
    return val * tl.view(
        (w * ((iy >= 0) * (iy < IH) * (ix >= 0) * (ix < IW)).to(tl.float32))[:, None]
        * channel_bcast,
        (BLOCK_SIZE, C),
    )


@triton.jit
def _grid_sample(
    image,
    batch_index,
    ix,
    iy,
    N: tl.constexpr,
    C: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    ix = ((ix + 1) / 2) * IW - 0.5
    iy = ((iy + 1) / 2) * IH - 0.5

    ix_nw = (ix - ix % 1).to(tl.float32)  # floor
    iy_nw = (iy - iy % 1).to(tl.float32)  # floor

    ix_ne = ix_nw + 1
    iy_ne = iy_nw
    ix_sw = ix_nw
    iy_sw = iy_nw + 1
    ix_se = ix_nw + 1
    iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    out_val = (
        _sample_2d(image, nw, batch_index, ix_nw, iy_nw, IH, IW, C, BLOCK_SIZE)
        + _sample_2d(image, ne, batch_index, ix_ne, iy_ne, IH, IW, C, BLOCK_SIZE)
        + _sample_2d(image, se, batch_index, ix_se, iy_se, IH, IW, C, BLOCK_SIZE)
        + _sample_2d(image, sw, batch_index, ix_sw, iy_sw, IH, IW, C, BLOCK_SIZE)
    )

    return out_val


@triton.jit()
def _load_mlp_weights(
    weight_1,
    bias_1,
    weight_2,
    bias_2,
    weight_3,
    bias_3,
    C: tl.constexpr,
    nn_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # MLP weights, biases
    w1 = tl.load(
        weight_1 + tl.arange(0, nn_dim)[:, None] * C + tl.arange(0, C)[None, :]
    )
    b1 = tl.load(bias_1 + tl.arange(0, nn_dim))
    b1 = tl.broadcast_to(b1.to(tl.float32)[None, :], (BLOCK_SIZE, nn_dim))

    w2 = tl.load(
        weight_2
        + tl.arange(0, nn_dim)[:, None] * nn_dim
        + tl.arange(0, nn_dim)[None, :]
    )
    b2 = tl.load(bias_2 + tl.arange(0, nn_dim))
    b2 = tl.broadcast_to(b2.to(tl.float32)[None, :], (BLOCK_SIZE, nn_dim))

    w3 = tl.load(
        weight_3 + tl.arange(0, C)[:, None] * nn_dim + tl.arange(0, nn_dim)[None, :]
    )
    b3 = tl.load(bias_3 + tl.arange(0, C))
    b3 = tl.broadcast_to(b3.to(tl.float32)[None, :], (BLOCK_SIZE, C))

    return w1, b1, w2, b2, w3, b3


@triton.jit
def _sample_3d(
    image,
    w,
    batch_index,
    ix,
    iy,
    iz,
    ID: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    channel_bcast = tl.full((1, C), 1.0, dtype=tl.float32)
    Coffs = tl.arange(0, C)
    ix_ = tl.minimum(tl.maximum(ix, 0.0), IW - 1).to(tl.int32)
    iy_ = tl.minimum(tl.maximum(iy, 0.0), IH - 1).to(tl.int32)
    iz_ = tl.minimum(tl.maximum(iz, 0.0), ID - 1).to(tl.int32)
    val = tl.view(
        tl.load(
            (
                image
                + batch_index * ID * IW * IH * C
                + iz_ * IW * IH * C
                + iy_ * IW * C
                + ix_ * C
            )[:, None]
            + Coffs[None, :]
        ),
        (BLOCK_SIZE, C),
    )
    return val * tl.view(
        (
            w
            * (
                (iy >= 0) * (iy < IH) * (ix >= 0) * (ix < IW) * (iz < ID) * (iz >= 0)
            ).to(tl.float32)
        )[:, None]
        * channel_bcast,
        (BLOCK_SIZE, C),
    )


@triton.jit
def _voxel_grid_sample(
    image,
    batch_index,
    ix,
    iy,
    iz,
    N: tl.constexpr,
    C: tl.constexpr,
    ID: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    ix = ((ix + 1) / 2) * IW - 0.5
    iy = ((iy + 1) / 2) * IH - 0.5
    iz = ((iz + 1) / 2) * ID - 0.5

    # ijk = xyz.astype(np.int32)
    # i, j, k = ijk[:,0], ijk[:,1], ijk[:,2]
    ix0 = (ix - ix % 1).to(tl.float32)  # floor
    iy0 = (iy - iy % 1).to(tl.float32)  # floor
    iz0 = (iz - iz % 1).to(tl.float32)  # floor

    # V000 = data[ i   , j   ,  k   ].astype(np.int32)
    # V100 = data[(i+1), j   ,  k   ].astype(np.int32)
    # V010 = data[ i   ,(j+1),  k   ].astype(np.int32)
    # V001 = data[ i   , j   , (k+1)].astype(np.int32)
    # V101 = data[(i+1), j   , (k+1)].astype(np.int32)
    # V011 = data[ i   ,(j+1), (k+1)].astype(np.int32)
    # V110 = data[(i+1),(j+1),  k   ].astype(np.int32)
    # V111 = data[(i+1),(j+1), (k+1)].astype(np.int32)

    V000x = ix0
    V000y = iy0
    V000z = iz0

    V100x = ix0
    V100y = iy0
    V100z = iz0 + 1

    V010x = ix0
    V010y = iy0 + 1
    V010z = iz0

    V001x = ix0 + 1
    V001y = iy0
    V001z = iz0

    V101x = ix0 + 1
    V101y = iy0
    V101z = iz0 + 1

    V011x = ix0 + 1
    V011y = iy0 + 1
    V011z = iz0

    V110x = ix0
    V110y = iy0 + 1
    V110z = iz0 + 1

    V111x = ix0 + 1
    V111y = iy0 + 1
    V111z = iz0 + 1

    # xyz = xyz - ijk

    x = ix - ix0
    y = iy - iy0
    z = iz - iz0

    # Vxyz = (V000 * (1 - x)*(1 - y)*(1 - z)
    #         + V100 * x * (1 - y) * (1 - z) +
    #         + V010 * (1 - x) * y * (1 - z) +
    #         + V001 * (1 - x) * (1 - y) * z +
    #         + V101 * x * (1 - y) * z +
    #         + V011 * (1 - x) * y * z +
    #         + V110 * x * y * (1 - z) +
    #         + V111 * x * y * z)

    out_val = (
        _sample_3d(
            image,
            (1 - x) * (1 - y) * (1 - z),
            batch_index,
            V000x,
            V000y,
            V000z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_3d(
            image,
            (1 - x) * (1 - y) * z,
            batch_index,
            V100x,
            V100y,
            V100z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_3d(
            image,
            (1 - x) * y * (1 - z),
            batch_index,
            V010x,
            V010y,
            V010z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_3d(
            image,
            x * (1 - y) * (1 - z),
            batch_index,
            V001x,
            V001y,
            V001z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_3d(
            image,
            x * (1 - y) * z,
            batch_index,
            V101x,
            V101y,
            V101z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_3d(
            image,
            x * y * (1 - z),
            batch_index,
            V011x,
            V011y,
            V011z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_3d(
            image,
            (1 - x) * y * z,
            batch_index,
            V110x,
            V110y,
            V110z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_3d(
            image,
            x * y * z,
            batch_index,
            V111x,
            V111y,
            V111z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
    )

    return out_val


@triton.jit
def _sample_grid_rep(
    xy,
    yz,
    zx,
    xt,
    yt,
    zt,
    batch_index,
    sample_x,
    sample_y,
    sample_z,
    sample_t,
    batch_size: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    shape_representation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    if shape_representation == 1:
        vec = _voxel_grid_sample(
            xy,
            batch_index,
            sample_x,
            sample_y,
            sample_z,
            batch_size,
            C,
            D,
            H,
            W,
            BLOCK_SIZE,
        )
    elif shape_representation == 0:
        a = _grid_sample(
            xy, batch_index, sample_x, sample_y, batch_size, C, H, W, BLOCK_SIZE
        )
        b = _grid_sample(
            yz, batch_index, sample_y, sample_z, batch_size, C, D, H, BLOCK_SIZE
        )
        c = _grid_sample(
            zx, batch_index, sample_z, sample_x, batch_size, C, W, D, BLOCK_SIZE
        )
        vec = a + b + c
    elif shape_representation == 2:
        a = _grid_sample(
            xy, batch_index, sample_x, sample_y, batch_size, C, H, W, BLOCK_SIZE
        )
        b = _grid_sample(
            yz, batch_index, sample_y, sample_z, batch_size, C, D, H, BLOCK_SIZE
        )
        c = _grid_sample(
            zx, batch_index, sample_z, sample_x, batch_size, C, W, D, BLOCK_SIZE
        )

        d = _grid_sample(
            xt, batch_index, sample_x, sample_t, batch_size, C, T, W, BLOCK_SIZE
        )
        e = _grid_sample(
            yt, batch_index, sample_y, sample_t, batch_size, C, T, H, BLOCK_SIZE
        )
        f = _grid_sample(
            zt, batch_index, sample_z, sample_t, batch_size, C, T, D, BLOCK_SIZE
        )
        vec = a + b + c + d + e + f
    vec = tl.view(vec, (BLOCK_SIZE, C))
    return vec


@triton.jit
def _splat_2d(
    to_splat,
    grad_image,
    w,
    batch_index,
    ix,
    iy,
    IH: tl.constexpr,
    IW: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    channel_bcast = tl.full((1, C), 1.0, dtype=tl.float32)
    Coffs = tl.arange(0, C)
    ix_ = tl.minimum(tl.maximum(ix, 0.0), IW - 1).to(tl.int32)
    iy_ = tl.minimum(tl.maximum(iy, 0.0), IH - 1).to(tl.int32)
    w = tl.view(
        (w * ((iy >= 0) * (iy < IH) * (ix >= 0) * (ix < IW)).to(tl.float32))[:, None]
        * channel_bcast,
        (BLOCK_SIZE, C),
    )
    offs = tl.view(
        (batch_index * IW * IH * C + iy_ * IW * C + ix_ * C)[:, None] + Coffs[None, :],
        (BLOCK_SIZE, C),
    )
    tl.atomic_add(grad_image + offs, w * to_splat)


@triton.jit
def _grid_splat(
    to_splat,
    feature_img,
    batch_index,
    ix,
    iy,
    N: tl.constexpr,
    C: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    ix = ((ix + 1) / 2) * IW - 0.5
    iy = ((iy + 1) / 2) * IH - 0.5

    ix_nw = (ix - ix % 1).to(tl.float32)  # floor
    iy_nw = (iy - iy % 1).to(tl.float32)  # floor

    ix_ne = ix_nw + 1
    iy_ne = iy_nw
    ix_sw = ix_nw
    iy_sw = iy_nw + 1
    ix_se = ix_nw + 1
    iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    _splat_2d(
        to_splat, feature_img, nw, batch_index, ix_nw, iy_nw, IH, IW, C, BLOCK_SIZE
    )
    _splat_2d(
        to_splat, feature_img, ne, batch_index, ix_ne, iy_ne, IH, IW, C, BLOCK_SIZE
    )
    _splat_2d(
        to_splat, feature_img, sw, batch_index, ix_sw, iy_sw, IH, IW, C, BLOCK_SIZE
    )
    _splat_2d(
        to_splat, feature_img, se, batch_index, ix_se, iy_se, IH, IW, C, BLOCK_SIZE
    )


@triton.jit
def _splat_3d(
    to_splat,
    grad_image,
    w,
    batch_index,
    ix,
    iy,
    iz,
    ID: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    channel_bcast = tl.full((1, C), 1.0, dtype=tl.float32)
    Coffs = tl.arange(0, C)
    ix_ = tl.minimum(tl.maximum(ix, 0.0), IW - 1).to(tl.int32)
    iy_ = tl.minimum(tl.maximum(iy, 0.0), IH - 1).to(tl.int32)
    iz_ = tl.minimum(tl.maximum(iz, 0.0), ID - 1).to(tl.int32)
    w = tl.view(
        w
        * ((iy >= 0) * (iy < IH) * (ix >= 0) * (ix < IW) * (iz < ID) * (iz >= 0)).to(
            tl.float32
        )[:, None]
        * channel_bcast,
        (BLOCK_SIZE, C),
    )
    offs = tl.view(
        (batch_index * ID * IW * IH * C + iz_ * IW * IH * C + iy_ * IW * C + ix_ * C)[
            :, None
        ]
        + Coffs[None, :],
        (BLOCK_SIZE, C),
    )
    tl.atomic_add(grad_image + offs, w * to_splat)


@triton.jit
def _voxel_grid_splat(
    to_splat,
    grad_image,
    batch_index,
    ix,
    iy,
    iz,
    N: tl.constexpr,
    C: tl.constexpr,
    ID: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # TODO: use this instead:
    #   - https://github.com/NVlabs/stylegan3/blob/407db86e6fe432540a22515310188288687858fa/torch_utils/ops/grid_sample_gradfix.py

    ix = ((ix + 1) / 2) * IW - 0.5
    iy = ((iy + 1) / 2) * IH - 0.5
    iz = ((iz + 1) / 2) * ID - 0.5

    # ijk = xyz.astype(np.int32)
    # i, j, k = ijk[:,0], ijk[:,1], ijk[:,2]

    ix0 = (ix - ix % 1).to(tl.float32)  # floor
    iy0 = (iy - iy % 1).to(tl.float32)  # floor
    iz0 = (iz - iz % 1).to(tl.float32)  # floor

    # V000 = data[ i   , j   ,  k   ].astype(np.int32)
    # V100 = data[(i+1), j   ,  k   ].astype(np.int32)
    # V010 = data[ i   ,(j+1),  k   ].astype(np.int32)
    # V001 = data[ i   , j   , (k+1)].astype(np.int32)
    # V101 = data[(i+1), j   , (k+1)].astype(np.int32)
    # V011 = data[ i   ,(j+1), (k+1)].astype(np.int32)
    # V110 = data[(i+1),(j+1),  k   ].astype(np.int32)
    # V111 = data[(i+1),(j+1), (k+1)].astype(np.int32)

    V000x = ix0
    V000y = iy0
    V000z = iz0

    V100x = ix0
    V100y = iy0
    V100z = iz0 + 1

    V010x = ix0
    V010y = iy0 + 1
    V010z = iz0

    V001x = ix0 + 1
    V001y = iy0
    V001z = iz0

    V101x = ix0 + 1
    V101y = iy0
    V101z = iz0 + 1

    V011x = ix0 + 1
    V011y = iy0 + 1
    V011z = iz0

    V110x = ix0
    V110y = iy0 + 1
    V110z = iz0 + 1

    V111x = ix0 + 1
    V111y = iy0 + 1
    V111z = iz0 + 1

    # xyz = xyz - ijk

    x = ix - ix0
    y = iy - iy0
    z = iz - iz0

    _splat_3d(
        to_splat,
        grad_image,
        (1 - x) * (1 - y) * (1 - z),
        batch_index,
        V000x,
        V000y,
        V000z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )
    _splat_3d(
        to_splat,
        grad_image,
        (1 - x) * (1 - y) * z,
        batch_index,
        V100x,
        V100y,
        V100z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )
    _splat_3d(
        to_splat,
        grad_image,
        (1 - x) * y * (1 - z),
        batch_index,
        V010x,
        V010y,
        V010z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )
    _splat_3d(
        to_splat,
        grad_image,
        x * (1 - y) * (1 - z),
        batch_index,
        V001x,
        V001y,
        V001z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )
    _splat_3d(
        to_splat,
        grad_image,
        x * (1 - y) * z,
        batch_index,
        V101x,
        V101y,
        V101z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )
    _splat_3d(
        to_splat,
        grad_image,
        x * y * (1 - z),
        batch_index,
        V011x,
        V011y,
        V011z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )
    _splat_3d(
        to_splat,
        grad_image,
        (1 - x) * y * z,
        batch_index,
        V110x,
        V110y,
        V110z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )
    _splat_3d(
        to_splat,
        grad_image,
        x * y * z,
        batch_index,
        V111x,
        V111y,
        V111z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )


@triton.jit
def _splat_grid_rep(
    to_splat,
    xy,
    yz,
    zx,
    xt,
    yt,
    zt,
    batch_index,
    sample_x,
    sample_y,
    sample_z,
    sample_t,
    batch_size: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    shape_representation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    if shape_representation == 0:
        _grid_splat(
            to_splat,
            xy,
            batch_index,
            sample_x,
            sample_y,
            batch_size,
            C,
            H,
            W,
            BLOCK_SIZE,
        )
        _grid_splat(
            to_splat,
            yz,
            batch_index,
            sample_y,
            sample_z,
            batch_size,
            C,
            D,
            H,
            BLOCK_SIZE,
        )
        _grid_splat(
            to_splat,
            zx,
            batch_index,
            sample_z,
            sample_x,
            batch_size,
            C,
            W,
            D,
            BLOCK_SIZE,
        )
    elif shape_representation == 1:
        _voxel_grid_splat(
            to_splat,
            xy,
            batch_index,
            sample_x,
            sample_y,
            sample_z,
            batch_size,
            C,
            D,
            H,
            W,
            BLOCK_SIZE,
        )
    elif shape_representation == 2:
        _grid_splat(
            to_splat,
            xy,
            batch_index,
            sample_x,
            sample_y,
            batch_size,
            C,
            H,
            W,
            BLOCK_SIZE,
        )
        _grid_splat(
            to_splat,
            yz,
            batch_index,
            sample_y,
            sample_z,
            batch_size,
            C,
            D,
            H,
            BLOCK_SIZE,
        )
        _grid_splat(
            to_splat,
            zx,
            batch_index,
            sample_z,
            sample_x,
            batch_size,
            C,
            W,
            D,
            BLOCK_SIZE,
        )
        _grid_splat(
            to_splat,
            xt,
            batch_index,
            sample_x,
            sample_t,
            batch_size,
            C,
            T,
            W,
            BLOCK_SIZE,
        )
        _grid_splat(
            to_splat,
            yt,
            batch_index,
            sample_y,
            sample_t,
            batch_size,
            C,
            T,
            H,
            BLOCK_SIZE,
        )
        _grid_splat(
            to_splat,
            zt,
            batch_index,
            sample_z,
            sample_t,
            batch_size,
            C,
            T,
            D,
            BLOCK_SIZE,
        )
    return 0


@triton.jit
def _d_linear_relu(d_y, w, b, xwb, x):
    # gradients of `y = max(x @ w + b, 0); xwb = x @ w + b`
    d_y_relu = d_y * (xwb > 0.0).to(tl.float32)
    return _d_linear(d_y_relu, w, b, x)


@triton.jit
def _d_linear(d_y, w, b, x):
    # gradients of `y = x @ w + b;
    d_x = tl.dot(d_y, tl.trans(w), allow_tf32=ALLOW_TF32)
    d_w = tl.dot(tl.trans(d_y), x, allow_tf32=ALLOW_TF32)
    d_b = tl.sum(d_y, axis=0)
    return d_x, d_w, d_b


@triton.jit
def _is_in_bounds(
    x,
    y,
    z,
    W: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    in_bounds = (
        (y >= -1.0) * (y <= 1.0) * (x >= -1.0) * (x <= 1.0) * (z >= -1.0) * (z <= 1.0)
    )
    in_bounds_mask = tl.broadcast_to(in_bounds.to(tl.float32)[:, None], (BLOCK_SIZE, C))
    return in_bounds_mask


@triton.jit
def _fw_kernel(
    H: tl.constexpr,
    W: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    C: tl.constexpr,
    xy_plane,
    yz_plane,
    zx_plane,
    xt_plane,
    yt_plane,
    zt_plane,
    img_feat,
    rays,
    centers,
    times,
    near,
    far,
    num_samples: tl.constexpr,
    num_samples_inf: tl.constexpr,
    num_rays: tl.constexpr,
    batch_size: tl.constexpr,
    mask,
    contract_coords: tl.constexpr,
    disparity_at_inf: tl.constexpr,
    cc_bound: tl.constexpr,
    shape_representation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_rays_total = num_rays * batch_size
    tot_num_samples = num_samples + num_samples_inf

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    offs_x = pid * BLOCK_SIZE * 3 + tl.arange(0, BLOCK_SIZE) * 3
    offs_y = offs_x + 1
    offs_z = offs_y + 1

    offs_t = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    offs_features = (
        pid * BLOCK_SIZE * C
        + C * tl.arange(0, BLOCK_SIZE)[:, None]
        + tl.arange(0, C)[None, :]
    )  # [BLOCK_SIZE, C]
    offs_features_mask = (
        pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    ) < num_rays_total
    center_x = tl.load(centers + offs_x, mask=offs_x < num_rays_total * 3).to(
        tl.float32
    )
    center_y = tl.load(centers + offs_y, mask=offs_y < num_rays_total * 3).to(
        tl.float32
    )
    center_z = tl.load(centers + offs_z, mask=offs_z < num_rays_total * 3).to(
        tl.float32
    )

    ray_x = tl.load(rays + offs_x, mask=offs_x < num_rays_total * 3).to(tl.float32)
    ray_y = tl.load(rays + offs_y, mask=offs_y < num_rays_total * 3).to(tl.float32)
    ray_z = tl.load(rays + offs_z, mask=offs_z < num_rays_total * 3).to(tl.float32)

    sample_t = tl.load(times + offs_t, mask=offs_t < num_rays).to(tl.float32)

    batch_index = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) // num_rays

    near_buffer = tl.load(near + offs, mask=offs < num_rays_total).to(tl.float32)
    far_buffer = tl.load(far + offs, mask=offs < num_rays_total).to(tl.float32)

    depth = near_buffer

    feature = tl.load(img_feat + offs_features, mask=offs_features_mask).to(tl.float32)
    mask = tl.load(mask + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
    mask = tl.broadcast_to(mask.to(tl.float32)[:, None], (BLOCK_SIZE, C))
    for step in range(tot_num_samples):
        if step < num_samples:
            depth = _depth_lin(near_buffer, far_buffer, num_samples, step)
        else:
            depth = _depth_inv_sphere(
                far_buffer,
                disparity_at_inf,
                num_samples_inf,
                step - num_samples,
            )

        sample_x = center_x + depth * ray_x
        sample_y = center_y + depth * ray_y
        sample_z = center_z + depth * ray_z

        if contract_coords:
            sample_x, sample_y, sample_z = _contract_pi(
                sample_x, sample_y, sample_z, cc_bound
            )

        in_bounds_mask = _is_in_bounds(
            sample_x, sample_y, sample_z, W, H, D, C, BLOCK_SIZE
        )
        feature_buffer = feature * in_bounds_mask

        feature_buffer = feature_buffer * mask

        _splat_grid_rep(
            feature_buffer,
            xy_plane,
            yz_plane,
            zx_plane,
            xt_plane,
            yt_plane,
            zt_plane,
            batch_index,
            sample_x,
            sample_y,
            sample_z,
            sample_t,
            batch_size,
            C,
            T,
            D,
            H,
            W,
            shape_representation,
            BLOCK_SIZE,
        )


@triton.jit
def _fw_recursive_kernel(
    H: tl.constexpr,
    W: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    C: tl.constexpr,
    nn_dim: tl.constexpr,
    xy_plane_prev,
    yz_plane_prev,
    zx_plane_prev,
    xt_plane_prev,
    yt_plane_prev,
    zt_plane_prev,
    xy_plane,
    yz_plane,
    zx_plane,
    xt_plane,
    yt_plane,
    zt_plane,
    weight_1,
    bias_1,
    weight_2,
    bias_2,
    weight_3,
    bias_3,
    img_feat,
    rays,
    centers,
    times,
    near,
    far,
    num_samples: tl.constexpr,
    num_samples_inf: tl.constexpr,
    num_rays: tl.constexpr,
    batch_size: tl.constexpr,
    mask,
    contract_coords: tl.constexpr,
    disparity_at_inf: tl.constexpr,
    cc_bound: tl.constexpr,
    shape_representation: tl.constexpr,
    use_multiply: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    debug_tensor,
):
    # shape_representation:
    #   0: triplane
    #   1: voxel grid, yz, zx are ignored
    #   2: HexPlane, all things about t are ignored
    pid = tl.program_id(axis=0)
    num_rays_total = num_rays * batch_size
    tot_num_samples = num_samples + num_samples_inf

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    offs_x = pid * BLOCK_SIZE * 3 + tl.arange(0, BLOCK_SIZE) * 3
    offs_y = offs_x + 1
    offs_z = offs_y + 1

    offs_t = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    offs_features = (
        pid * BLOCK_SIZE * C
        + C * tl.arange(0, BLOCK_SIZE)[:, None]
        + tl.arange(0, C)[None, :]
    )  # [BLOCK_SIZE, C]
    offs_features_mask = (
        pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    ) < num_rays_total

    center_x = tl.load(centers + offs_x, mask=offs_x < num_rays_total * 3).to(
        tl.float32
    )
    center_y = tl.load(centers + offs_y, mask=offs_y < num_rays_total * 3).to(
        tl.float32
    )
    center_z = tl.load(centers + offs_z, mask=offs_z < num_rays_total * 3).to(
        tl.float32
    )

    ray_x = tl.load(rays + offs_x, mask=offs_x < num_rays_total * 3).to(tl.float32)
    ray_y = tl.load(rays + offs_y, mask=offs_y < num_rays_total * 3).to(tl.float32)
    ray_z = tl.load(rays + offs_z, mask=offs_z < num_rays_total * 3).to(tl.float32)

    sample_t = tl.load(times + offs_t, mask=offs_t < num_rays_total).to(tl.float32)

    # tl.store(debug_tensor + offs_t, sample_t, mask=offs_t < num_rays_total)

    batch_index = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) // num_rays

    near_buffer = tl.load(near + offs, mask=offs < num_rays_total).to(tl.float32)
    far_buffer = tl.load(far + offs, mask=offs < num_rays_total).to(tl.float32)

    depth = near_buffer

    feature = tl.load(img_feat + offs_features, mask=offs_features_mask).to(tl.float32)
    mask = tl.load(mask + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
    mask = tl.broadcast_to(mask.to(tl.float32)[:, None], (BLOCK_SIZE, C))

    w1, b1, w2, b2, w3, b3 = _load_mlp_weights(
        weight_1, bias_1, weight_2, bias_2, weight_3, bias_3, C, nn_dim, BLOCK_SIZE
    )
    for step in range(tot_num_samples):
        if step < num_samples:
            depth = _depth_lin(near_buffer, far_buffer, num_samples, step)
        else:
            depth = _depth_inv_sphere(
                far_buffer,
                disparity_at_inf,
                num_samples_inf,
                step - num_samples,
            )

        sample_x = center_x + depth * ray_x
        sample_y = center_y + depth * ray_y
        sample_z = center_z + depth * ray_z

        if contract_coords:
            sample_x, sample_y, sample_z = _contract_pi(
                sample_x, sample_y, sample_z, cc_bound
            )

        vec = _sample_grid_rep(
            xy_plane_prev,
            yz_plane_prev,
            zx_plane_prev,
            xt_plane_prev,
            yt_plane_prev,
            zt_plane_prev,
            batch_index,
            sample_x,
            sample_y,
            sample_z,
            sample_t,
            batch_size,
            C,
            T,
            D,
            H,
            W,
            shape_representation,
            BLOCK_SIZE,
        )

        # tl.store(
        #     debug_tensor
        #     + tl.arange(0, BLOCK_SIZE)[:, None] * C
        #     + tl.arange(0, C)[None, :],
        #     vec,
        # )
        if use_multiply:
            feature_buffer = feature * vec
        else:
            feature_buffer = feature + vec

        feature_buffer = tl.maximum(
            tl.dot(feature_buffer, tl.trans(w1), allow_tf32=ALLOW_TF32) + b1, 0.0
        )
        feature_buffer = tl.maximum(
            tl.dot(feature_buffer, tl.trans(w2), allow_tf32=ALLOW_TF32) + b2, 0.0
        )
        feature_buffer = (
            tl.dot(feature_buffer, tl.trans(w3), allow_tf32=ALLOW_TF32) + b3
        )

        in_bounds_mask = _is_in_bounds(
            sample_x, sample_y, sample_z, W, H, D, C, BLOCK_SIZE
        )
        feature_buffer = feature_buffer * in_bounds_mask

        feature_buffer = feature_buffer * mask

        _splat_grid_rep(
            feature_buffer,
            xy_plane,
            yz_plane,
            zx_plane,
            xt_plane,
            yt_plane,
            zt_plane,
            batch_index,
            sample_x,
            sample_y,
            sample_z,
            sample_t,
            batch_size,
            C,
            T,
            D,
            H,
            W,
            shape_representation,
            BLOCK_SIZE,
        )


@triton.jit
def _bw_recursive_kernel(
    H: tl.constexpr,
    W: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    C: tl.constexpr,
    nn_dim: tl.constexpr,
    xy_plane_prev,
    yz_plane_prev,
    zx_plane_prev,
    xt_plane_prev,
    yt_plane_prev,
    zt_plane_prev,
    weight_1,
    bias_1,
    weight_2,
    bias_2,
    weight_3,
    bias_3,
    img_feat,
    grad_xy_plane_prev,
    grad_yz_plane_prev,
    grad_zx_plane_prev,
    grad_xt_plane_prev,
    grad_yt_plane_prev,
    grad_zt_plane_prev,
    grad_xy_plane,
    grad_yz_plane,
    grad_zx_plane,
    grad_xt_plane,
    grad_yt_plane,
    grad_zt_plane,
    grad_weight_1,
    grad_bias_1,
    grad_weight_2,
    grad_bias_2,
    grad_weight_3,
    grad_bias_3,
    grad_img_feat,
    rays,
    centers,
    times,
    near,
    far,
    num_samples,
    num_samples_inf: tl.constexpr,
    num_rays: tl.constexpr,
    batch_size: tl.constexpr,
    mask,
    contract_coords: tl.constexpr,
    disparity_at_inf: tl.constexpr,
    cc_bound: tl.constexpr,
    shape_representation: tl.constexpr,
    use_multiply: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    debug_tensor,
):
    pid = tl.program_id(axis=0)
    num_rays_total = num_rays * batch_size
    tot_num_samples = num_samples + num_samples_inf

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    offs_x = pid * BLOCK_SIZE * 3 + tl.arange(0, BLOCK_SIZE) * 3
    offs_y = offs_x + 1
    offs_z = offs_y + 1

    offs_t = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    offs_features = (
        pid * BLOCK_SIZE * C
        + C * tl.arange(0, BLOCK_SIZE)[:, None]
        + tl.arange(0, C)[None, :]
    )  # [BLOCK_SIZE, C]
    offs_features_mask = (
        pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    ) < num_rays_total
    center_x = tl.load(centers + offs_x, mask=offs_x < num_rays_total * 3).to(
        tl.float32
    )
    center_y = tl.load(centers + offs_y, mask=offs_y < num_rays_total * 3).to(
        tl.float32
    )
    center_z = tl.load(centers + offs_z, mask=offs_z < num_rays_total * 3).to(
        tl.float32
    )

    ray_x = tl.load(rays + offs_x, mask=offs_x < num_rays_total * 3).to(tl.float32)
    ray_y = tl.load(rays + offs_y, mask=offs_y < num_rays_total * 3).to(tl.float32)
    ray_z = tl.load(rays + offs_z, mask=offs_z < num_rays_total * 3).to(tl.float32)

    sample_t = tl.load(times + offs_t, mask=offs_t < num_rays_total).to(tl.float32)

    batch_index = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) // num_rays

    near_buffer = tl.load(near + offs, mask=offs < num_rays_total).to(tl.float32)
    far_buffer = tl.load(far + offs, mask=offs < num_rays_total).to(tl.float32)

    depth = near_buffer

    grad_img_buffer = tl.zeros((BLOCK_SIZE, C), dtype=tl.float32)
    mask = tl.load(mask + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.float32)
    w1, b1, w2, b2, w3, b3 = _load_mlp_weights(
        weight_1, bias_1, weight_2, bias_2, weight_3, bias_3, C, nn_dim, BLOCK_SIZE
    )

    d_w1 = tl.zeros((nn_dim, C), dtype=tl.float32)
    d_b1 = tl.zeros((nn_dim,), dtype=tl.float32)
    d_w2 = tl.zeros((nn_dim, nn_dim), dtype=tl.float32)
    d_b2 = tl.zeros((nn_dim,), dtype=tl.float32)
    d_w3 = tl.zeros((C, nn_dim), dtype=tl.float32)
    d_b3 = tl.zeros((C,), dtype=tl.float32)
    feature = tl.load(img_feat + offs_features, mask=offs_features_mask).to(tl.float32)

    for step in range(tot_num_samples):
        if step < num_samples:
            depth = _depth_lin(near_buffer, far_buffer, num_samples, step)
        else:
            depth = _depth_inv_sphere(
                far_buffer,
                disparity_at_inf,
                num_samples_inf,
                step - num_samples,
            )

        sample_x = center_x + depth * ray_x
        sample_y = center_y + depth * ray_y
        sample_z = center_z + depth * ray_z

        if contract_coords:
            sample_x, sample_y, sample_z = _contract_pi(
                sample_x, sample_y, sample_z, cc_bound
            )

        grad_vec = _sample_grid_rep(
            grad_xy_plane,
            grad_yz_plane,
            grad_zx_plane,
            grad_xt_plane,
            grad_yt_plane,
            grad_zt_plane,
            batch_index,
            sample_x,
            sample_y,
            sample_z,
            sample_t,
            batch_size,
            C,
            T,
            D,
            H,
            W,
            shape_representation,
            BLOCK_SIZE,
        )

        vec = _sample_grid_rep(
            xy_plane_prev,
            yz_plane_prev,
            zx_plane_prev,
            xt_plane_prev,
            yt_plane_prev,
            zt_plane_prev,
            batch_index,
            sample_x,
            sample_y,
            sample_z,
            sample_t,
            batch_size,
            C,
            T,
            D,
            H,
            W,
            shape_representation,
            BLOCK_SIZE,
        )
        if use_multiply:
            feature_buffer = feature * vec
        else:
            feature_buffer = feature + vec

        feature_1 = tl.maximum(
            tl.dot(feature_buffer, tl.trans(w1), allow_tf32=ALLOW_TF32) + b1, 0.0
        )
        feature_2 = tl.maximum(
            tl.dot(feature_1, tl.trans(w2), allow_tf32=ALLOW_TF32) + b2, 0.0
        )
        # feature_3 = tl.dot(feature_2, tl.trans(w3), allow_tf32=ALLOW_TF32) + b3

        in_bounds_mask = _is_in_bounds(
            sample_x, sample_y, sample_z, W, H, D, C, BLOCK_SIZE
        )
        grad_vec = grad_vec * in_bounds_mask
        grad_vec = grad_vec * mask[:, None]

        d_x_2, d_w_3, d_b_3 = _d_linear(grad_vec, tl.trans(w3), b3, feature_2)
        d_x_1, d_w_2, d_b_2 = _d_linear_relu(
            d_x_2, tl.trans(w2), b2, feature_2, feature_1
        )
        d_x_0, d_w_1, d_b_1 = _d_linear_relu(
            d_x_1, tl.trans(w1), b1, feature_1, feature_buffer
        )

        d_w1 = d_w1 + d_w_1
        d_b1 = d_b1 + d_b_1
        d_w2 = d_w2 + d_w_2
        d_b2 = d_b2 + d_b_2
        d_w3 = d_w3 + d_w_3
        d_b3 = d_b3 + d_b_3
        tl.store(
            debug_tensor
            + tl.arange(0, BLOCK_SIZE)[:, None] * nn_dim
            + tl.arange(0, nn_dim)[None, :],
            d_x_1,
        )

        # NOTE: this line might be related to this bug: https://github.com/openai/triton/issues/1864
        d_x_0 = d_x_0 + grad_vec * 0.0
        # remove above line will fail the compile
        # so stupid.
        if use_multiply:
            d_x_0_image = d_x_0 * vec
            d_x_0_plane = d_x_0 * feature
        else:
            d_x_0_image = d_x_0
            d_x_0_plane = d_x_0
        _splat_grid_rep(
            d_x_0_plane,
            grad_xy_plane_prev,
            grad_yz_plane_prev,
            grad_zx_plane_prev,
            grad_xt_plane_prev,
            grad_yt_plane_prev,
            grad_zt_plane_prev,
            batch_index,
            sample_x,
            sample_y,
            sample_z,
            sample_t,
            batch_size,
            C,
            T,
            D,
            H,
            W,
            shape_representation,
            BLOCK_SIZE,
        )
        grad_img_buffer = grad_img_buffer + d_x_0_image

    tl.store(grad_img_feat + offs_features, grad_img_buffer, mask=offs_features_mask)
    tl.atomic_add(
        grad_weight_1 + tl.arange(0, nn_dim)[:, None] * C + tl.arange(0, C)[None, :],
        d_w1,
    )
    tl.atomic_add(grad_bias_1 + tl.arange(0, nn_dim), d_b1)
    tl.atomic_add(
        grad_weight_2
        + tl.arange(0, nn_dim)[:, None] * nn_dim
        + tl.arange(0, nn_dim)[None, :],
        d_w2,
    )
    tl.atomic_add(grad_bias_2 + tl.arange(0, nn_dim), d_b2)
    tl.atomic_add(
        grad_weight_3
        + tl.arange(0, C)[:, None] * nn_dim
        + tl.arange(0, nn_dim)[None, :],
        d_w3,
    )
    tl.atomic_add(grad_bias_3 + tl.arange(0, C), d_b3)


@triton.jit
def _bw_kernel(
    H: tl.constexpr,
    W: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    C: tl.constexpr,
    grad_xy_plane,
    grad_yz_plane,
    grad_zx_plane,
    grad_xt_plane,
    grad_yt_plane,
    grad_zt_plane,
    grad_img_feat,
    rays,
    centers,
    times,
    near,
    far,
    num_samples,
    num_samples_inf: tl.constexpr,
    num_rays: tl.constexpr,
    batch_size: tl.constexpr,
    mask,
    contract_coords: tl.constexpr,
    disparity_at_inf: tl.constexpr,
    cc_bound: tl.constexpr,
    shape_representation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    debug_tensor,
):
    pid = tl.program_id(axis=0)
    num_rays_total = num_rays * batch_size
    tot_num_samples = num_samples + num_samples_inf

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    offs_x = pid * BLOCK_SIZE * 3 + tl.arange(0, BLOCK_SIZE) * 3
    offs_y = offs_x + 1
    offs_z = offs_y + 1

    offs_features = (
        pid * BLOCK_SIZE * C
        + C * tl.arange(0, BLOCK_SIZE)[:, None]
        + tl.arange(0, C)[None, :]
    )  # [BLOCK_SIZE, C]
    offs_features_mask = (
        pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    ) < num_rays_total

    offs_t = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    center_x = tl.load(centers + offs_x, mask=offs_x < num_rays_total * 3).to(
        tl.float32
    )
    center_y = tl.load(centers + offs_y, mask=offs_y < num_rays_total * 3).to(
        tl.float32
    )
    center_z = tl.load(centers + offs_z, mask=offs_z < num_rays_total * 3).to(
        tl.float32
    )

    ray_x = tl.load(rays + offs_x, mask=offs_x < num_rays_total * 3).to(tl.float32)
    ray_y = tl.load(rays + offs_y, mask=offs_y < num_rays_total * 3).to(tl.float32)
    ray_z = tl.load(rays + offs_z, mask=offs_z < num_rays_total * 3).to(tl.float32)

    sample_t = tl.load(times + offs_t, mask=offs_t < num_rays_total).to(tl.float32)

    batch_index = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) // num_rays

    near_buffer = tl.load(near + offs, mask=offs < num_rays_total).to(tl.float32)
    far_buffer = tl.load(far + offs, mask=offs < num_rays_total).to(tl.float32)

    depth = near_buffer

    grad_img_buffer = tl.zeros((BLOCK_SIZE, C), dtype=tl.float32)
    mask = tl.load(mask + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.float32)

    for step in range(tot_num_samples):
        if step < num_samples:
            depth = _depth_lin(near_buffer, far_buffer, num_samples, step)
        else:
            depth = _depth_inv_sphere(
                far_buffer,
                disparity_at_inf,
                num_samples_inf,
                step - num_samples,
            )

        sample_x = center_x + depth * ray_x
        sample_y = center_y + depth * ray_y
        sample_z = center_z + depth * ray_z

        if contract_coords:
            sample_x, sample_y, sample_z = _contract_pi(
                sample_x, sample_y, sample_z, cc_bound
            )

        grad_vec = _sample_grid_rep(
            grad_xy_plane,
            grad_yz_plane,
            grad_zx_plane,
            grad_xt_plane,
            grad_yt_plane,
            grad_zt_plane,
            batch_index,
            sample_x,
            sample_y,
            sample_z,
            sample_t,
            batch_size,
            C,
            T,
            D,
            H,
            W,
            shape_representation,
            BLOCK_SIZE,
        )

        in_bounds_mask = _is_in_bounds(
            sample_x, sample_y, sample_z, W, H, D, C, BLOCK_SIZE
        )
        grad_vec = grad_vec * in_bounds_mask
        grad_vec = grad_vec * mask[:, None]
        grad_img_buffer = grad_img_buffer + grad_vec

    tl.store(grad_img_feat + offs_features, grad_img_buffer, mask=offs_features_mask)
