// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// BLAKE2b's compression function for Metal, as blocks/hash/blake2b_f.h has it (RFC 7693).
#pragma once
constant uchar B2_SIGMA[12][16] = {
  {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},{14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3},
  {11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4},{7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8},
  {9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13},{2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9},
  {12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11},{13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10},
  {6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5},{10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0},
  {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},{14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3}};
constant ulong B2_IV[8] = {0x6a09e667f3bcc908ul,0xbb67ae8584caa73bul,0x3c6ef372fe94f82bul,0xa54ff53a5f1d36f1ul,
                           0x510e527fade682d1ul,0x9b05688c2b3e6c1ful,0x1f83d9abfb41bd6bul,0x5be0cd19137e2179ul};
inline ulong b2_rotr(ulong x, uint n) { return (x >> n) | (x << (64 - n)); }
#define B2_G(r,i,a,b,c,d) \
  a = a + b + m[B2_SIGMA[r][2*i]]; d = b2_rotr(d ^ a, 32); c = c + d; b = b2_rotr(b ^ c, 24); \
  a = a + b + m[B2_SIGMA[r][2*i+1]]; d = b2_rotr(d ^ a, 16); c = c + d; b = b2_rotr(b ^ c, 63);
inline void b2_compress(thread ulong* h, thread const ulong* m, ulong t_low, ulong t_high, bool last) {
  ulong v[16];
  for (int i = 0; i < 8; i++) { v[i] = h[i]; v[i + 8] = B2_IV[i]; }
  v[12] ^= t_low; v[13] ^= t_high; if (last) v[14] = ~v[14];
  for (int r = 0; r < 12; r++) {
    B2_G(r,0,v[0],v[4],v[8],v[12]) B2_G(r,1,v[1],v[5],v[9],v[13]) B2_G(r,2,v[2],v[6],v[10],v[14]) B2_G(r,3,v[3],v[7],v[11],v[15])
    B2_G(r,4,v[0],v[5],v[10],v[15]) B2_G(r,5,v[1],v[6],v[11],v[12]) B2_G(r,6,v[2],v[7],v[8],v[13]) B2_G(r,7,v[3],v[4],v[9],v[14])
  }
  for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
}
// The first chaining value for a digest of 32 bytes; start: BLAKE2b's salt and personalization.
inline void b2_begin(thread ulong* h, thread const ulong* start) {
  h[0] = B2_IV[0] ^ 0x01010020ul; h[1] = B2_IV[1]; h[2] = B2_IV[2]; h[3] = B2_IV[3];
  for (int i = 0; i < 4; i++) h[4 + i] = B2_IV[4 + i] ^ start[i];
}
