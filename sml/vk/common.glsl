// Shared declarations for the Sheep Vulkan kernels (sml-vk.c). Every kernel sees
// the same three buffers: S, the scratch (activations, the quantized activation,
// rope tables, logits); W, the weights of the stage it runs (one layer, or the
// head); KV, the key and value cache of every layer this device holds. Offsets
// arrive in push constants, in 32-bit words.
//
// The formats are the CUDA path's (sml-gpu-fwd.cu): flat 4-bit codes as a nibble
// plane (byte j of a 64-weight block: low nibble = weight j of group 2b, high =
// weight j of group 2b+1), 6-bit as that plane plus a 2-bit group-planar plane
// (code = lo4 | hi2 << 4), per-group uint8 scale and min fields under an fp16
// superblock scale per 8 groups, mins biased by 127.


layout(std430, set = 0, binding = 0) buffer SF { float sf[]; };
layout(std430, set = 0, binding = 0) buffer SU { uint su[]; };
layout(std430, set = 0, binding = 1) readonly buffer WU { uint wu[]; };
layout(std430, set = 0, binding = 1) readonly buffer WF { float wf[]; };
layout(std430, set = 0, binding = 2) buffer KVF { float kv[]; };
// 16-byte views for the matvecs' code and activation reads (planes start 16-byte aligned)
layout(std430, set = 0, binding = 0) buffer SU4 { uvec4 su4[]; };
layout(std430, set = 0, binding = 1) readonly buffer WU4 { uvec4 wu4[]; };

#define GROUP 32u
#define GPS 8u

// byte i of the byte array that starts at word `base` of W
uint wbyte(uint base, uint i) { return (wu[base + (i >> 2)] >> ((i & 3u) * 8u)) & 0xFFu; }

// fp16 element i of the half array that starts at word `base` of W
float whalf(uint base, uint i) {
    uint w = wu[base + (i >> 1)];
    return unpackHalf2x16((i & 1u) == 0u ? w : (w >> 16)).x;
}

// signed byte b of a word of int8 activations
int sb(uint w, uint b) { return bitfieldExtract(int(w), int(b * 8u), 8); }
