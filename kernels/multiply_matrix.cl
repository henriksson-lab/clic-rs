__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#define LM_STRIDE (TILE_SIZE + 1)

#ifndef WPT_M
  #define WPT_M 1
#endif

#define RTS_M (TILE_SIZE / WPT_M)

__kernel void multiply_matrix(
    IMAGE_src0_TYPE  src0,
    IMAGE_src1_TYPE  src1,
    IMAGE_dst_TYPE   dst
)
{
  const int col = get_global_id(0);
  const int local_x = get_local_id(0);
  const int local_y = get_local_id(1);

  const int M = GET_IMAGE_HEIGHT(src0);
  const int K = GET_IMAGE_WIDTH(src0);
  const int N = GET_IMAGE_WIDTH(src1);

  const int row_base = get_group_id(1) * TILE_SIZE + local_y;

  // tile_src0 is stored TRANSPOSED: [k][m] instead of [m][k]
  // This means the compute loop reads sequential memory addresses
  // when iterating over the m (row) dimension within a tile.
  __local float tile_src0[TILE_SIZE][LM_STRIDE];
  __local float tile_src1[TILE_SIZE][LM_STRIDE];

  float acc[WPT_M];
  #pragma unroll
  for (int w = 0; w < WPT_M; ++w)
    acc[w] = 0.0f;

  const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (int tile = 0; tile < num_tiles; ++tile)
  {
    const int tile_offset = tile * TILE_SIZE;

    #pragma unroll
    for (int w = 0; w < WPT_M; ++w)
    {
      const int load_row = local_y + w * RTS_M;
      const int a_row = get_group_id(1) * TILE_SIZE + load_row;
      const int a_col = tile_offset + local_x;

      // TRANSPOSED store: swap indices so A is stored as [k][m]
      // local_x corresponds to the k dimension within the tile
      // load_row corresponds to the m dimension within the tile
      if (a_row < M && a_col < K)
        tile_src0[local_x][load_row] = READ_IMAGE(src0, sampler, POS_src0_INSTANCE(a_col, a_row, 0, 0)).x;
      else
        tile_src0[local_x][load_row] = 0.0f;

      // B tile unchanged: stored as [k][n]
      const int b_row = tile_offset + load_row;
      const int b_col = col;

      if (b_row < K && b_col < N)
        tile_src1[load_row][local_x] = READ_IMAGE(src1, sampler, POS_src1_INSTANCE(b_col, b_row, 0, 0)).x;
      else
        tile_src1[load_row][local_x] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k)
    {
      const float b_val = tile_src1[k][local_x];

      #pragma unroll
      for (int w = 0; w < WPT_M; ++w)
      {
        // TRANSPOSED read: tile_src0[k][m] instead of tile_src0[m][k]
        // Adjacent threads (different w values) now read adjacent memory
        // addresses tile_src0[k][local_y], tile_src0[k][local_y+RTS_M], ...
        // which are consecutive in the row, eliminating bank conflicts.
        acc[w] += tile_src0[k][local_y + w * RTS_M] * b_val;
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  #pragma unroll
  for (int w = 0; w < WPT_M; ++w)
  {
    const int out_row = row_base + w * RTS_M;
    if (out_row < M && col < N)
    {
      WRITE_IMAGE(dst, POS_dst_INSTANCE(col, out_row, 0, 0), CONVERT_dst_PIXEL_TYPE(acc[w]));
    }
  }
}
