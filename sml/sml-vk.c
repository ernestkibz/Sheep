// sml-vk: Sheep models on any Vulkan GPU (AMD, Intel, NVIDIA), from the file's own
// 4- and 6-bit codes (Ernest, 2026-10-14: "more = better" across GPUs).
//
// The DirectML worker held every weight in fp16, 3.6x the file, so a 2 GB Radeon
// held 2 of a 7B's 28 layers and multi ran slower than hybrid. This holds the codes
// as the CUDA path does (sml-gpu-fwd.c up_tensor) and runs the same arithmetic
// (vk/*.comp, ported from sml-gpu-fwd.cu): per-group int8 activations against
// nibble / 2-bit planes under the two-level scales.
//
// It is a device of sml-server through the existing worker protocol (--ext-cmd;
// sml-server.c ext_open): hello 'SXEF', then prompt / step / bye / middle / front.
//
//   sml-vk --list                                   the Vulkan devices
//   sml-vk MODEL.ssf --grade [--device N] [--layers 0:B]   against the CPU engine
//   sml-vk MODEL.ssf --connect PORT [--device N] [--layers A:B] [--ctx N]
//
// Dense pre-norm families only for now (Llama, Mistral, Qwen2): anything else is
// refused with the reason, and the server stays on the CPU.

#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <windows.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sml-kernel.h"
#include "sml-model.h"
#include "sml-run.h"

#define MAGIC 0x46455853u     // 'SXEF'
// positions recorded into one submission: a prompt is one wait, not one per token
#define BATCH 64
// rows the kernels run at once (NBMAX in vk/mv4.comp): the weights are read once per NB rows
#define NB 8

// ------------------------------------------------------------------ Vulkan entry points
#define VKF(X) \
    X(vkCreateInstance) X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) \
    X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkCreateDevice) \
    X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) \
    X(vkBindBufferMemory) X(vkMapMemory) X(vkCreateShaderModule) X(vkCreateDescriptorSetLayout) \
    X(vkCreatePipelineLayout) X(vkCreateComputePipelines) X(vkCreateDescriptorPool) X(vkAllocateDescriptorSets) \
    X(vkUpdateDescriptorSets) X(vkCreateCommandPool) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) \
    X(vkEndCommandBuffer) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) \
    X(vkCmdDispatch) X(vkCmdPipelineBarrier) X(vkCmdCopyBuffer) X(vkCmdFillBuffer) X(vkQueueSubmit) \
    X(vkCreateFence) X(vkWaitForFences) X(vkResetFences) X(vkResetCommandBuffer) X(vkFreeMemory) \
    X(vkDestroyBuffer)
#define DECL(f) static PFN_##f f;
VKF(DECL)
static PFN_vkGetInstanceProcAddr gipa;
static VkInstance inst;

static int vk_load(void) {
    HMODULE h = LoadLibraryA("vulkan-1.dll");
    if (!h) return 0;
    gipa = (PFN_vkGetInstanceProcAddr) (void *) GetProcAddress(h, "vkGetInstanceProcAddr");
    if (!gipa) return 0;
    vkCreateInstance = (PFN_vkCreateInstance) gipa(NULL, "vkCreateInstance");
    if (!vkCreateInstance) return 0;
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "sml-vk", 1, "sheep", 1, VK_API_VERSION_1_1 };
    VkInstanceCreateInfo ci = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, 0, NULL };
    if (vkCreateInstance(&ci, NULL, &inst) != VK_SUCCESS) return 0;
#define LOADF(f) f = (PFN_##f) gipa(inst, #f); if (!f) { fprintf(stderr, "no %s\n", #f); return 0; }
    VKF(LOADF)
    return 1;
}

// ------------------------------------------------------------------ device
static VkPhysicalDevice phys;
static VkDevice dev;
static VkQueue queue;
static uint32_t qfam;
static VkPhysicalDeviceMemoryProperties memp;
static VkCommandPool cpool;
static VkCommandBuffer cb;
static VkFence fence;
static char dev_name[256];

static int list_devices(int pick, int print) {
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, NULL);
    VkPhysicalDevice pd[16];
    if (n > 16) n = 16;
    vkEnumeratePhysicalDevices(inst, &n, pd);
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties pr;
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceProperties(pd[i], &pr);
        vkGetPhysicalDeviceMemoryProperties(pd[i], &mp);
        double local = 0, host = 0;
        for (uint32_t h = 0; h < mp.memoryHeapCount; h++) {
            if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) local += (double) mp.memoryHeaps[h].size;
            else host += (double) mp.memoryHeaps[h].size;
        }
        if (print)
            printf("%u|%s|%04x|%d|%.0f|%.0f\n", i, pr.deviceName, pr.vendorID, (int) pr.deviceType,
                   local / 1048576.0, host / 1048576.0);
        if ((int) i == pick) { phys = pd[i]; snprintf(dev_name, sizeof dev_name, "%s", pr.deviceName); }
    }
    return (int) n;
}

static int open_device(int pick) {
    if (list_devices(pick, 0) <= pick || !phys) return 0;
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, NULL);
    VkQueueFamilyProperties qp[16];
    if (nq > 16) nq = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qp);
    qfam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }
    if (qfam == UINT32_MAX) return 0;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qfam, 1, &prio };
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &qci, 0, NULL, 0, NULL, NULL };
    if (vkCreateDevice(phys, &dci, NULL, &dev) != VK_SUCCESS) return 0;
    vkGetDeviceQueue(dev, qfam, 0, &queue);
    vkGetPhysicalDeviceMemoryProperties(phys, &memp);
    VkCommandPoolCreateInfo pci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL,
                                    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qfam };
    vkCreateCommandPool(dev, &pci, NULL, &cpool);
    VkCommandBufferAllocateInfo ai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cpool,
                                       VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
    vkAllocateCommandBuffers(dev, &ai, &cb);
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0 };
    vkCreateFence(dev, &fci, NULL, &fence);
    return 1;
}

// ------------------------------------------------------------------ buffers
typedef struct { VkBuffer buf; VkDeviceMemory mem; void * map; VkDeviceSize size; } vbuf;

// Device-local first (mappable if the driver offers it), then any mappable memory:
// an integrated GPU's carve-out is small, its shared memory large.
static int vb_alloc(vbuf * b, VkDeviceSize size, int host_only) {
    memset(b, 0, sizeof *b);
    b->size = size;
    VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL };
    if (vkCreateBuffer(dev, &bi, NULL, &b->buf) != VK_SUCCESS) return 0;
    VkMemoryRequirements rq;
    vkGetBufferMemoryRequirements(dev, b->buf, &rq);
    const VkMemoryPropertyFlags HV = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryPropertyFlags want[3] = { VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | HV,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, HV };
    for (int w = host_only ? 2 : 0; w < 3; w++)
        for (uint32_t t = 0; t < memp.memoryTypeCount; t++) {
            if (!(rq.memoryTypeBits & (1u << t))) continue;
            if ((memp.memoryTypes[t].propertyFlags & want[w]) != want[w]) continue;
            VkMemoryAllocateInfo ma = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, t };
            if (vkAllocateMemory(dev, &ma, NULL, &b->mem) != VK_SUCCESS) continue;
            vkBindBufferMemory(dev, b->buf, b->mem, 0);
            if (memp.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
                vkMapMemory(dev, b->mem, 0, VK_WHOLE_SIZE, 0, &b->map);
            return 1;
        }
    return 0;
}

static vbuf staging;

static void submit_wait(void) {
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb, 0, NULL };
    vkResetFences(dev, 1, &fence);
    vkQueueSubmit(queue, 1, &si, fence);
    vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
}

static void begin(void) {
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL,
                                    VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL };
    vkBeginCommandBuffer(cb, &bi);
}

static void upload(vbuf * b, VkDeviceSize at, const void * src, size_t n) {
    if (b->map) { memcpy((char *) b->map + at, src, n); return; }
    for (size_t done = 0; done < n; ) {
        size_t k = n - done < staging.size ? n - done : (size_t) staging.size;
        memcpy(staging.map, (const char *) src + done, k);
        begin();
        VkBufferCopy cp = { 0, at + done, k };
        vkCmdCopyBuffer(cb, staging.buf, b->buf, 1, &cp);
        vkEndCommandBuffer(cb);
        submit_wait();
        done += k;
    }
}

// ------------------------------------------------------------------ a stage's bytes
typedef struct { uint8_t * p; size_t n, cap; } blob;

static uint32_t put(blob * b, const void * src, size_t bytes) {
    size_t at = (b->n + 15) & ~(size_t) 15;          // 16-byte aligned: the matvecs read uvec4
    if (at + bytes + 4 > b->cap) {
        b->cap = (at + bytes + 4) * 2;
        b->p = realloc(b->p, b->cap);
    }
    memset(b->p + b->n, 0, at - b->n);
    if (src) memcpy(b->p + at, src, bytes); else memset(b->p + at, 0, bytes);
    b->n = at + bytes;
    return (uint32_t) (at / 4);
}

typedef struct { uint32_t c2, c4, sd, sq, md, mq, perm; int rows, n, six, have_min; const int32_t * hperm; } vt;

static sml_model M;
static sml_ctx C;

// As up_tensor (sml-gpu-fwd.c): the planes as they are; v9's 6-bit group fields widened
// to bytes with the min re-biased to 127, so the kernels read one format.
static vt put_tensor(blob * b, const sml_rec * r) {
    vt t;
    memset(&t, 0, sizeof t);
    if (!r) return t;
    t.rows = (int) r->rows; t.n = (int) r->cols; t.six = r->nbits == 6; t.have_min = r->mq_n ? 1 : 0;
    const size_t rows = (size_t) t.rows, n = (size_t) t.n, ng = n / SML_GROUP, nscale = (size_t) SML_NSCALE(t.n);
    if (t.six) {
        t.c2 = put(b, sml_ptr(&M, r->codes_off[0]), rows * (n / 128) * 32);
        t.c4 = put(b, sml_ptr(&M, r->codes_off[1]), rows * n / 2);
    } else {
        t.c4 = put(b, sml_ptr(&M, r->codes_off[0]), rows * n / 2);
    }
    t.sd = put(b, sml_ptr(&M, r->sd_off), rows * nscale * 2);
    if (sml_kernel_scale6_get()) {
        const uint8_t * s6 = sml_ptr(&M, r->sq_off);
        const uint8_t * m6 = t.have_min ? sml_ptr(&M, r->mq_off) : NULL;
        const size_t g6 = (size_t) SML_SCALE6_BYTES(ng);
        uint8_t * s8 = malloc(rows * ng), * m8 = t.have_min ? malloc(rows * ng) : NULL;
        for (size_t row = 0; row < rows; row++)
            for (size_t g = 0; g < ng; g++) {
                s8[row * ng + g] = (uint8_t) sml_scale6_at(s6 + row * g6, (int) g);
                if (m8) m8[row * ng + g] = (uint8_t) (sml_scale6_at(m6 + row * g6, (int) g)
                                                      + (sml_kernel_minneg_get() ? 64 : 96));
            }
        t.sq = put(b, s8, rows * ng);
        if (m8) { t.md = put(b, sml_ptr(&M, r->md_off), rows * nscale * 2); t.mq = put(b, m8, rows * ng); }
        free(s8); free(m8);
    } else {
        t.sq = put(b, sml_ptr(&M, r->sq_off), rows * ng);
        if (t.have_min) { t.md = put(b, sml_ptr(&M, r->md_off), rows * nscale * 2); t.mq = put(b, sml_ptr(&M, r->mq_off), rows * ng); }
    }
    if (!t.have_min) { t.md = t.sd; t.mq = t.sq; }
    t.hperm = (const int32_t *) sml_ptr(&M, r->perm_off);
    t.perm = put(b, t.hperm, n * 4);
    return t;
}

static uint32_t put_f(blob * b, const float * p, size_t n) { return p ? put(b, p, n * 4) : 0; }
static int same_perm(const vt * a, const vt * b) {
    return a->n && a->n == b->n && memcmp(a->hperm, b->hperm, (size_t) a->n * 4) == 0;
}

typedef struct {
    vt q, k, v, o, gate, up, down;
    uint32_t an, fn, qb, kb, vb;
    int kv_same, gu_same;
    vbuf w;
    VkDescriptorSet ds;
} vlayer;

// ------------------------------------------------------------------ model on the card
static int n_embd, n_head, n_kv, hd, kvd, n_ff, n_vocab, NL, half_, interleaved, n_ctx, wide;
static float eps_, inv_;
static int L0, L1, want_head;              // the stage: layers L0..L1-1, plus the head if want_head
static vlayer * VL;
static vt HEAD;
static uint32_t ON;                        // the output norm, in the head's buffer
static vbuf SB, KVB, IO, HW;               // scratch, cache, host window, the head's weights
static VkDescriptorSet DS_HEAD;
static uint32_t oX, oXB, oQ, oK, oV, oHB, oHB2, oLOG, oQW, oQS, oQM, oATT, oRC, oRS;
static uint32_t sX, sXB, sQ, sK, sHB, sHB2, sQW, sQS;      // row strides of the batched buffers, in words
static float * rowb;

enum { K_ADDNORM, K_QUANTW, K_MV4, K_MV6, K_ROPEKV, K_ATTN, K_SILU, K_ADD, K_MV4B, K_MV6B, K_N };
static const char * KNAME[K_N] = { "addnorm", "quantw", "mv4", "mv6", "ropekv", "attn", "silu", "add", "mv4b", "mv6b" };
static VkPipeline PIPE[K_N];
static VkPipelineLayout PL;
static VkDescriptorSetLayout DSL;
static VkDescriptorPool DPOOL;

static int supported(char * why, size_t nwhy) {
    const sml_hdr * h = M.hdr;
    const char * w = NULL;
    if (h->flags & SML_MODEL_HADAMARD) w = "Hadamard-rotated files";
    else if (h->version >= 4 && h->norm_type == 1) w = "LayerNorm models";
    else if (h->version >= 4 && (h->arch_bits & ~(uint32_t) 0)) w = "this architecture's options (GELU, ALiBi, parallel residual ...)";
    else if (h->emb_scale != 0.0f || h->res_scale != 0.0f || h->logit_scale != 0.0f || h->logit_softcap != 0.0f
             || h->attn_softcap != 0.0f || h->clip_qkv != 0.0f) w = "scaled or capped models";
    else if (h->n_swa || C.layer_win || C.no_rope || C.pos_embd || C.alibi || C.emb_norm || C.out_b) w = "windowed or position-table models";
    else if (h->version >= 4 && h->attn_scale != 0.0f) w = "a custom attention scale";
    for (int l = 0; !w && l < (int) h->n_layer; l++) {
        const sml_layer * s = &C.layers[l];
        if (s->n_exp || s->shg || s->ssm_in || s->m1_in || s->gdn_qkvz || s->sc_in || s->attn_gate || s->xielu)
            w = "experts or state-space layers";
        else if (!s->q || !s->gate || !s->attn_norm || !s->ffn_norm) w = "a layer without attention, gate or norms";
        else if (s->attn_post_norm || s->ffn_post_norm || s->q_norm || s->k_norm) w = "post-norms or Q/K norms";
        else if (s->o_bias || s->up_bias || s->down_bias || s->gate_bias) w = "biases beyond Q/K/V";
        else {
            const sml_rec * ts[7] = { s->q, s->k, s->v, s->o, s->gate, s->up, s->down };
            for (int t = 0; t < 7 && !w; t++)
                if (ts[t]->layout != SML_LAYOUT_FLAT || (ts[t]->nbits != 4 && ts[t]->nbits != 6)) w = "tensors other than flat 4- or 6-bit";
        }
    }
    if (!w && (C.head->layout != SML_LAYOUT_FLAT || (C.head->nbits != 4 && C.head->nbits != 6))) w = "a head other than flat 4- or 6-bit";
    if (w) { snprintf(why, nwhy, "not on Vulkan yet: %s", w); return 0; }
    return 1;
}

static int make_pipelines(const char * dir) {
    VkDescriptorSetLayoutBinding bd[3];
    for (int i = 0; i < 3; i++) {
        bd[i] = (VkDescriptorSetLayoutBinding) { (uint32_t) i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL };
    }
    VkDescriptorSetLayoutCreateInfo dl = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 3, bd };
    vkCreateDescriptorSetLayout(dev, &dl, NULL, &DSL);
    VkPushConstantRange pr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 128 };
    VkPipelineLayoutCreateInfo pl = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &DSL, 1, &pr };
    vkCreatePipelineLayout(dev, &pl, NULL, &PL);
    for (int k = 0; k < K_N; k++) {
        char path[1024];
        snprintf(path, sizeof path, "%s\\vk\\%s.spv", dir, KNAME[k]);
        FILE * f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "missing %s\n", path); return 0; }
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        uint32_t * code = malloc((size_t) n);
        if (fread(code, 1, (size_t) n, f) != (size_t) n) { fclose(f); return 0; }
        fclose(f);
        VkShaderModuleCreateInfo si = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, (size_t) n, code };
        VkShaderModule sm;
        if (vkCreateShaderModule(dev, &si, NULL, &sm) != VK_SUCCESS) return 0;
        VkComputePipelineCreateInfo ci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
            { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
            PL, VK_NULL_HANDLE, 0 };
        if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, NULL, &PIPE[k]) != VK_SUCCESS) return 0;
        free(code);
    }
    return 1;
}

static VkDescriptorSet make_set(vbuf * w) {
    VkDescriptorSet ds;
    VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, DPOOL, 1, &DSL };
    vkAllocateDescriptorSets(dev, &ai, &ds);
    VkDescriptorBufferInfo bi[3] = { { SB.buf, 0, VK_WHOLE_SIZE }, { w->buf, 0, VK_WHOLE_SIZE }, { KVB.buf, 0, VK_WHOLE_SIZE } };
    VkWriteDescriptorSet wr[3];
    for (int i = 0; i < 3; i++)
        wr[i] = (VkWriteDescriptorSet) { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, (uint32_t) i, 0, 1,
                                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi[i], NULL };
    vkUpdateDescriptorSets(dev, 3, wr, 0, NULL);
    return ds;
}

static int upload_blob(vbuf * w, blob * b) {
    if (!vb_alloc(w, b->n + 16, 0)) return 0;
    upload(w, 0, b->p, b->n);
    return 1;
}

static int load(const char * path, int ctx, int threads, int l0, int l1, int head, char * why, size_t nwhy) {
    if (!sml_model_open(&M, path)) { snprintf(why, nwhy, "cannot open the model"); return 0; }
    if (!sml_ctx_init(&C, &M, ctx, threads)) { snprintf(why, nwhy, "cannot set up the model"); return 0; }
    if (!supported(why, nwhy)) return 0;
    const sml_hdr * h = M.hdr;
    n_embd = (int) h->n_embd; n_head = (int) h->n_head; n_kv = (int) h->n_head_kv; n_ff = (int) h->n_ff;
    n_vocab = (int) h->n_vocab; NL = (int) h->n_layer; hd = C.head_dim; kvd = n_kv * hd;
    half_ = C.rope_half; interleaved = h->version >= 4 && h->rope_type == 1; n_ctx = ctx; eps_ = h->rms_eps;
    inv_ = 1.0f / sqrtf((float) hd);
    L0 = l0; L1 = l1 > 0 ? l1 : NL; want_head = head;
    if (L0 < 0 || L0 >= L1 || L1 > NL) { snprintf(why, nwhy, "layers %d..%d outside the model's %d", L0, L1, NL); return 0; }
    wide = n_ff > n_embd ? n_ff : n_embd;
    if (n_head * hd > wide) wide = n_head * hd;
    if (!vb_alloc(&staging, 64u << 20, 1)) { snprintf(why, nwhy, "no host-visible memory"); return 0; }
    // scratch layout, in words
    uint32_t at = 0;
#define PAD(n) ((uint32_t) (((n) + 63) & ~63))
#define TAKE(o, s, n) do { s = PAD(n); o = at; at += s * NB; } while (0)
    uint32_t one;
    TAKE(oX, sX, n_embd); TAKE(oXB, sXB, wide); TAKE(oQ, sQ, n_head * hd); TAKE(oK, sK, kvd); TAKE(oV, sK, kvd);
    TAKE(oHB, sHB, n_ff); TAKE(oHB2, sHB2, wide); TAKE(oQW, sQW, wide / 4); TAKE(oQS, sQS, wide / 32);
    TAKE(oQM, sQS, wide / 32); TAKE(oATT, one, (size_t) n_head * n_ctx);
    oLOG = at; at += PAD(n_vocab);
    oRC = at; at += PAD((size_t) n_ctx * half_);
    oRS = at; at += PAD((size_t) n_ctx * half_);
    if (!vb_alloc(&SB, (VkDeviceSize) at * 4, 0)) { snprintf(why, nwhy, "no memory for the scratch"); return 0; }
    upload(&SB, (VkDeviceSize) oRC * 4, C.rope_cos, (size_t) n_ctx * half_ * 4);
    upload(&SB, (VkDeviceSize) oRS * 4, C.rope_sin, (size_t) n_ctx * half_ * 4);
    const int nl = L1 - L0;
    if (!vb_alloc(&KVB, (VkDeviceSize) 2 * nl * n_ctx * kvd * 4, 0)) { snprintf(why, nwhy, "no memory for the cache"); return 0; }
    if (!vb_alloc(&IO, ((VkDeviceSize) 2 * BATCH * n_embd + (VkDeviceSize) n_vocab) * 4 + 64, 1)) { snprintf(why, nwhy, "no host window"); return 0; }
    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t) (3 * (nl + 2)) };
    VkDescriptorPoolCreateInfo pi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, (uint32_t) (nl + 2), 1, &ps };
    vkCreateDescriptorPool(dev, &pi, NULL, &DPOOL);
    VL = calloc((size_t) nl, sizeof(vlayer));
    blob b = { 0 };
    for (int l = L0; l < L1; l++) {
        const sml_layer * s = &C.layers[l];
        vlayer * y = &VL[l - L0];
        b.n = 0;
        y->q = put_tensor(&b, s->q); y->k = put_tensor(&b, s->k); y->v = put_tensor(&b, s->v); y->o = put_tensor(&b, s->o);
        y->gate = put_tensor(&b, s->gate); y->up = put_tensor(&b, s->up); y->down = put_tensor(&b, s->down);
        y->an = put_f(&b, (const float *) sml_ptr(&M, s->attn_norm->off), (size_t) n_embd);
        y->fn = put_f(&b, (const float *) sml_ptr(&M, s->ffn_norm->off), (size_t) n_embd);
        y->qb = s->q_bias ? put_f(&b, s->q_bias, (size_t) n_head * hd) : 0;
        y->kb = s->k_bias ? put_f(&b, s->k_bias, (size_t) kvd) : 0;
        y->vb = s->v_bias ? put_f(&b, s->v_bias, (size_t) kvd) : 0;
        y->kv_same = same_perm(&y->q, &y->k) && same_perm(&y->q, &y->v);
        y->gu_same = same_perm(&y->gate, &y->up);
        if (!upload_blob(&y->w, &b)) { snprintf(why, nwhy, "out of GPU memory at layer %d", l); return 0; }
        y->ds = make_set(&y->w);
    }
    if (want_head) {
        b.n = 0;
        HEAD = put_tensor(&b, C.head);
        ON = put_f(&b, (const float *) sml_ptr(&M, C.out_norm->off), (size_t) n_embd);
        if (!upload_blob(&HW, &b)) { snprintf(why, nwhy, "out of GPU memory for the head"); return 0; }
        DS_HEAD = make_set(&HW);
    } else {
        DS_HEAD = VL[nl - 1].ds;
    }
    free(b.p);
    rowb = malloc((size_t) n_embd * 4);
    return 1;
}

// ------------------------------------------------------------------ recording a token
static void barrier(void) {
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL,
                           VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

static int g_skip = -1;     // SML_VK_SKIP=k: leave kernel k out (profiling only: the answer is wrong)
static void run(int k, VkDescriptorSet ds, const void * pc, uint32_t size, uint32_t gx, uint32_t gy) {
    if (k == g_skip || (g_skip == 99 && (k == K_MV4 || k == K_MV6 || k == K_MV4B || k == K_MV6B))) return;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, PIPE[k]);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, PL, 0, 1, &ds, 0, NULL);
    vkCmdPushConstants(cb, PL, VK_SHADER_STAGE_COMPUTE_BIT, 0, size, pc);
    vkCmdDispatch(cb, gx ? gx : 1, gy ? gy : 1, 1);
    barrier();
}
static uint32_t cap(uint32_t g) { return g > 65535u ? 65535u : (g ? g : 1u); }

// nb rows of x (stride sx) normed into out (stride so); mode 1 adds y (stride sy) first
static void addnorm(VkDescriptorSet ds, uint32_t x, uint32_t sx, uint32_t y, uint32_t sy, uint32_t w, int mode, int nb) {
    struct { uint32_t x, y, out, w, b, n, mode, hb; float eps; uint32_t sx, sy, so; } p =
        { x, y, oXB, w, 0, (uint32_t) n_embd, (uint32_t) mode, 0, eps_, sx, sy, sXB };
    run(K_ADDNORM, ds, &p, sizeof p, 1, (uint32_t) nb);
}
static void quantw(VkDescriptorSet ds, uint32_t x, uint32_t sx, const vt * t, int nb) {
    struct { uint32_t x, perm, qw, qs, qm, n, sx, sqw, sqs; } p = { x, t->perm, oQW, oQS, oQM, (uint32_t) t->n, sx, sQW, sQS };
    run(K_QUANTW, ds, &p, sizeof p, cap((uint32_t) (t->n / 32 + 63) / 64), (uint32_t) nb);
}
static void mv(VkDescriptorSet ds, const vt * t, uint32_t y, uint32_t sy, uint32_t bias, int nb) {
    if (t->six) {
        struct { uint32_t y, c2, c4, sd, sq, md, mq, qw, qs, qm, bias, rows, n, hm, hb, nb, sy, sqw, sqs; } p =
            { y, t->c2, t->c4, t->sd, t->sq, t->md, t->mq, oQW, oQS, oQM, bias, (uint32_t) t->rows, (uint32_t) t->n,
              (uint32_t) t->have_min, bias ? 1u : 0u, (uint32_t) nb, sy, sQW, sQS };
        run(nb > 1 ? K_MV6B : K_MV6, ds, &p, sizeof p, cap((uint32_t) (t->rows + 3) / 4), 1);   // one row: the leaner kernel
    } else {
        struct { uint32_t y, c4, sd, sq, md, mq, qw, qs, qm, bias, rows, n, hm, hb, nb, sy, sqw, sqs; } p =
            { y, t->c4, t->sd, t->sq, t->md, t->mq, oQW, oQS, oQM, bias, (uint32_t) t->rows, (uint32_t) t->n,
              (uint32_t) t->have_min, bias ? 1u : 0u, (uint32_t) nb, sy, sQW, sQS };
        run(nb > 1 ? K_MV4B : K_MV4, ds, &p, sizeof p, cap((uint32_t) (t->rows + 3) / 4), 1);
    }
}

// one layer for nb rows at positions pos0 .. pos0 + nb - 1
static void rec_layer(int l, int pos0, int first, int nb) {
    const vlayer * y = &VL[l - L0];
    VkDescriptorSet ds = y->ds;
    addnorm(ds, oX, sX, oHB2, sHB2, y->an, first ? 0 : 1, nb);
    quantw(ds, oXB, sXB, &y->q, nb); mv(ds, &y->q, oQ, sQ, y->qb, nb);
    if (!y->kv_same) quantw(ds, oXB, sXB, &y->k, nb);
    mv(ds, &y->k, oK, sK, y->kb, nb);
    if (!y->kv_same) quantw(ds, oXB, sXB, &y->v, nb);
    mv(ds, &y->v, oV, sK, y->vb, nb);
    const uint32_t lk = (uint32_t) ((size_t) (l - L0) * n_ctx * kvd);
    const uint32_t lv = (uint32_t) ((size_t) (L1 - L0) * n_ctx * kvd) + lk;
    {
        struct { uint32_t q, k, v, kc, vc, cs, sn, nh, nk, hd, half, il, sq, sk, kvd; } p =
            { oQ, oK, oV, lk + (uint32_t) pos0 * kvd, lv + (uint32_t) pos0 * kvd, oRC + (uint32_t) pos0 * half_,
              oRS + (uint32_t) pos0 * half_, (uint32_t) n_head, (uint32_t) n_kv, (uint32_t) hd, (uint32_t) half_,
              (uint32_t) interleaved, sQ, sK, (uint32_t) kvd };
        run(K_ROPEKV, ds, &p, sizeof p, cap((uint32_t) (n_head + 63) / 64), (uint32_t) nb);
    }
    {
        struct { uint32_t out, q, kl, vl, att, nh, nk, hd, nc, t0, pos; float inv; uint32_t so, sq; } p =
            { oXB, oQ, lk, lv, oATT, (uint32_t) n_head, (uint32_t) n_kv, (uint32_t) hd, (uint32_t) n_ctx, 0,
              (uint32_t) pos0, inv_, sXB, sQ };
        run(K_ATTN, ds, &p, sizeof p, (uint32_t) n_head, (uint32_t) nb);
    }
    quantw(ds, oXB, sXB, &y->o, nb); mv(ds, &y->o, oHB2, sHB2, 0, nb);
    addnorm(ds, oX, sX, oHB2, sHB2, y->fn, 1, nb);
    quantw(ds, oXB, sXB, &y->gate, nb); mv(ds, &y->gate, oHB, sHB, 0, nb);
    if (!y->gu_same) quantw(ds, oXB, sXB, &y->up, nb);
    mv(ds, &y->up, oHB2, sHB2, 0, nb);
    {
        struct { uint32_t g, u, n, sg, su; } p = { oHB, oHB2, (uint32_t) n_ff, sHB, sHB2 };
        run(K_SILU, ds, &p, sizeof p, cap((uint32_t) (n_ff + 255) / 256), (uint32_t) nb);
    }
    quantw(ds, oHB, sHB, &y->down, nb); mv(ds, &y->down, oHB2, sHB2, 0, nb);
}

static void host_embed(int tok, float * out) {
    const sml_rec * e = C.emb;
    const uint16_t * esd = (const uint16_t *) sml_ptr(&M, e->sd_off);
    const uint8_t  * esq = sml_ptr(&M, e->sq_off);
    const uint16_t * emd = e->mq_n ? (const uint16_t *) sml_ptr(&M, e->md_off) : NULL;
    const uint8_t  * emq = e->mq_n ? sml_ptr(&M, e->mq_off) : NULL;
    if (e->nbits == 6)
        sml_dequant_row_flat6(rowb, sml_ptr(&M, e->codes_off[1]), sml_ptr(&M, e->codes_off[0]), esd, esq, emd, emq, tok, n_embd);
    else
        sml_dequant_row_flat4(rowb, sml_ptr(&M, e->codes_off[0]), esd, esq, emd, emq, tok, n_embd);
    const int32_t * ep = (const int32_t *) sml_ptr(&M, e->perm_off);
    for (int i = 0; i < n_embd; i++) out[ep[i]] = rowb[i];
}

// n positions through this stage in one submission, NB rows at a time (the weights are
// read once per NB rows). in: the tokens (when the stage embeds) or n residual rows.
// out: the last position's logits (with the head; the earlier positions only fill the
// cache, so the large head runs once), or the n rows after the stage.
static void steps(int n, const int * ids, const float * rows_in, int pos0, float * out) {
    for (int at = 0; at < n; at += BATCH) {
        const int k = n - at < BATCH ? n - at : BATCH;
        float * io = (float *) IO.map;
        float * outrows = io + (size_t) BATCH * n_embd;
        for (int i = 0; i < k; i++) {
            if (rows_in) memcpy(io + (size_t) i * n_embd, rows_in + (size_t) (at + i) * n_embd, (size_t) n_embd * 4);
            else host_embed(ids[at + i], io + (size_t) i * n_embd);
        }
        begin();
        VkDeviceSize bytes = 0;
        for (int i0 = 0; i0 < k; i0 += NB) {
            const int nb = k - i0 < NB ? k - i0 : NB;
            VkBufferCopy in[NB];
            for (int b = 0; b < nb; b++)
                in[b] = (VkBufferCopy) { (VkDeviceSize) (i0 + b) * n_embd * 4, ((VkDeviceSize) oX + (VkDeviceSize) b * sX) * 4,
                                         (VkDeviceSize) n_embd * 4 };
            vkCmdCopyBuffer(cb, IO.buf, SB.buf, (uint32_t) nb, in);
            barrier();
            for (int l = L0; l < L1; l++) rec_layer(l, pos0 + at + i0, l == L0, nb);
            if (want_head) {
                if (at + i0 + nb == n) {                       // the last row of the prompt only
                    const uint32_t r = (uint32_t) nb - 1;
                    addnorm(DS_HEAD, oX + r * sX, sX, oHB2 + r * sHB2, sHB2, ON, 1, 1);
                    quantw(DS_HEAD, oXB, sXB, &HEAD, 1); mv(DS_HEAD, &HEAD, oLOG, 0, 0, 1);
                    VkBufferCopy back = { (VkDeviceSize) oLOG * 4, (VkDeviceSize) BATCH * n_embd * 4, (VkDeviceSize) n_vocab * 4 };
                    vkCmdCopyBuffer(cb, SB.buf, IO.buf, 1, &back);
                    bytes = back.size;
                }
            } else {
                struct { uint32_t x, y, n, sx, sy; } p = { oX, oHB2, (uint32_t) n_embd, sX, sHB2 };
                run(K_ADD, VL[0].ds, &p, sizeof p, cap((uint32_t) (n_embd + 255) / 256), (uint32_t) nb);
                VkBufferCopy back[NB];
                for (int b = 0; b < nb; b++)
                    back[b] = (VkBufferCopy) { ((VkDeviceSize) oX + (VkDeviceSize) b * sX) * 4,
                                               ((VkDeviceSize) BATCH + i0 + b) * n_embd * 4, (VkDeviceSize) n_embd * 4 };
                vkCmdCopyBuffer(cb, SB.buf, IO.buf, (uint32_t) nb, back);
            }
            barrier();
        }
        vkEndCommandBuffer(cb);
        submit_wait();
        if (want_head) { if (bytes) memcpy(out, outrows, (size_t) bytes); }
        else memcpy(out + (size_t) at * n_embd, outrows, (size_t) k * n_embd * 4);
    }
}

static void step(int tok, const float * row_in, int pos, float * out) { steps(1, &tok, row_in, pos, out); }

// ------------------------------------------------------------------ grading against the CPU
static double corr(const float * a, const float * b, int n) {
    double ma = 0, mb = 0, sab = 0, saa = 0, sbb = 0;
    for (int i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    for (int i = 0; i < n; i++) { double x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
    return sab / sqrt(saa * sbb + 1e-30);
}
static int argmax(const float * v, int n) { int b = 0; for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i; return b; }

static int grade(int threads) {
    // fixed ids, then greedy continuation from the CPU's choice: the maths is compared, not words
    int ids[16] = { 100, 200, 300, 400, 500 }, n = 5, gen = 2;
    sml_ctx R, S;
    if (!sml_ctx_init(&R, &M, n_ctx, threads)) return 1;
    if (!want_head && !sml_ctx_init(&S, &M, n_ctx, threads)) return 1;
    float * ref = malloc((size_t) n_vocab * 4), * got = malloc((size_t) n_vocab * 4), * row = malloc((size_t) n_embd * 4);
    double worst = 1, sum = 0;
    int agree = 0, total = 0;
    double t_vk = 0;
    for (int i = 0; i < n + gen; i++) {
        memcpy(ref, sml_forward(&R, ids[i], i, 0), (size_t) n_vocab * 4);
        LARGE_INTEGER f, a, b; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&a);
        if (want_head) step(ids[i], NULL, i, got);
        else { step(ids[i], NULL, i, row); memcpy(got, sml_forward_from(&S, row, i, L1, 0), (size_t) n_vocab * 4); }
        QueryPerformanceCounter(&b); t_vk += (double) (b.QuadPart - a.QuadPart) / (double) f.QuadPart;
        double c = corr(ref, got, n_vocab);
        sum += c; if (c < worst) worst = c;
        int tr = argmax(ref, n_vocab), tg = argmax(got, n_vocab);
        if (tr == tg || ref[tg] >= ref[tr] - 0.01f * fabsf(ref[tr])) agree++;      // near-ties agree
        total++;
        if (i + 1 >= n && i + 1 < n + gen) ids[i + 1] = tr;
    }
    printf("  device %s, layers %d..%d%s\n", dev_name, L0, L1 - 1, want_head ? " + head" : " (the rest on the CPU)");
    printf("  GPU vs CPU over %d positions: mean logit corr %f, worst %f, top-1 agree %d/%d\n",
           total, sum / total, worst, agree, total);
    printf("  %.2f tok/s on this path (one position at a time)\n", total / t_vk);
    int pass = sum / total >= 0.99 && agree >= total - 1;
    if (want_head) {
        // a 12-token prompt in one call: the batched path (NB rows a pass) against the CPU's last logits
        int pr[12];
        sml_ctx R2;
        if (!sml_ctx_init(&R2, &M, n_ctx, threads)) return 1;
        for (int i = 0; i < 12; i++) { pr[i] = 100 + 37 * i; memcpy(ref, sml_forward(&R2, pr[i], i, 0), (size_t) n_vocab * 4); }
        LARGE_INTEGER f, a, b; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&a);
        steps(12, pr, NULL, 0, got);
        QueryPerformanceCounter(&b);
        const double c = corr(ref, got, n_vocab), s = (double) (b.QuadPart - a.QuadPart) / (double) f.QuadPart;
        const int same = argmax(ref, n_vocab) == argmax(got, n_vocab);
        printf("  batched prompt, 12 tokens in %d-row passes: corr %f, top-1 %s, %.1f tok/s\n",
               NB, c, same ? "same" : "different", 12 / s);
        pass = pass && c >= 0.99;
    }
    return pass ? 0 : 1;
}

// ------------------------------------------------------------------ the worker protocol
static SOCKET sk, nx = INVALID_SOCKET;      // the server (or the card before), the next card's worker
static int sendn(SOCKET s_, const void * p, int n) { const char * c = p; while (n > 0) { int k = send(s_, c, n, 0); if (k <= 0) return 0; c += k; n -= k; } return 1; }
static int recvn(SOCKET s_, void * p, int n) { char * c = p; while (n > 0) { int k = recv(s_, c, n, 0); if (k <= 0) return 0; c += k; n -= k; } return 1; }
static int send_all(const void * p, int n) { return sendn(sk, p, n); }
static int recv_all(void * p, int n) { return recvn(sk, p, n); }

// Several GPUs (--chain D:END,D:END...): this worker runs its layers on its card, then hands
// the rows to the next card's worker, which it starts itself (the same exe and model, the
// next layers); the server sees one worker. Hidden rows cross cards through the host.
static char next_name[256];
static int start_next(const char * model, int ctx, const char * chain, char * why, size_t nwhy) {
    int dev2 = 0, end2 = 0, used = 0;
    if (sscanf(chain, "%d:%d%n", &dev2, &end2, &used) < 2 || end2 <= L1) { snprintf(why, nwhy, "bad --chain %s", chain); return 0; }
    const char * rest = chain[used] == ',' ? chain + used + 1 : "";
    WSADATA wd;
    WSAStartup(MAKEWORD(2, 2), &wd);
    SOCKET ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    struct sockaddr_in a = { 0 };
    int alen = sizeof a;
    a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = 0;
    if (bind(ls, (struct sockaddr *) &a, sizeof a) || listen(ls, 1) || getsockname(ls, (struct sockaddr *) &a, &alen)) {
        snprintf(why, nwhy, "no local port for the next card"); return 0;
    }
    char exe[1024], cmd[4096];
    GetModuleFileNameA(NULL, exe, sizeof exe);
    snprintf(cmd, sizeof cmd, "\"%s\" \"%s\" --device %d --layers %d:%d --ctx %d%s%s --connect %d", exe, model, dev2, L1, end2,
             ctx, *rest ? " --chain " : "", rest, (int) ntohs(a.sin_port));
    STARTUPINFOA si = { sizeof si };
    PROCESS_INFORMATION pi;
    if (!CreateProcessA(NULL, cmd, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
        snprintf(why, nwhy, "could not start the next card's worker"); return 0;
    }
    nx = accept(ls, NULL, NULL);
    closesocket(ls);
    if (nx == INVALID_SOCKET) { snprintf(why, nwhy, "the next card's worker did not connect"); return 0; }
    int nodelay = 1;
    setsockopt(nx, IPPROTO_TCP, TCP_NODELAY, (const char *) &nodelay, sizeof nodelay);
    int32_t hello[4];
    if (!recvn(nx, hello, sizeof hello) || hello[0] != (int32_t) MAGIC || hello[3] < 0 || hello[3] > 255) {
        snprintf(why, nwhy, "the next card's worker did not answer"); return 0;
    }
    if (!recvn(nx, next_name, hello[3])) return 0;
    next_name[hello[3]] = 0;
    if (hello[1] != 0) { snprintf(why, nwhy, "next card: %s", next_name); return 0; }
    return 1;
}

// the rows after this card go through the rest of the chain
static int pass_on(int pos0, int n, float * rows) {
    if (nx == INVALID_SOCKET) return 1;
    int32_t hdr[3] = { 4, pos0, n }, status = 1;
    if (!sendn(nx, hdr, sizeof hdr) || !sendn(nx, rows, n * n_embd * 4) || !recvn(nx, &status, 4) || status != 0) return 0;
    return recvn(nx, rows, n * n_embd * 4);
}

static int serve(int port, int ok, const char * text) {
    WSADATA wd;
    WSAStartup(MAKEWORD(2, 2), &wd);
    sk = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    struct sockaddr_in a = { 0 };
    a.sin_family = AF_INET; a.sin_port = htons((u_short) port); a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(sk, (struct sockaddr *) &a, sizeof a) != 0) return 1;
    int nodelay = 1;
    setsockopt(sk, IPPROTO_TCP, TCP_NODELAY, (const char *) &nodelay, sizeof nodelay);
    char named[600];
    if (ok && next_name[0]) { snprintf(named, sizeof named, "%s + %s", text, next_name); text = named; }
    int32_t hello[4] = { (int32_t) MAGIC, ok ? 0 : 1, ok ? n_vocab : 0, (int32_t) strlen(text) };
    if (!send_all(hello, sizeof hello) || !send_all(text, hello[3])) return 1;
    if (!ok) return 0;
    float * out = malloc((size_t) (n_vocab > n_embd ? n_vocab : n_embd) * 4);
    for (;;) {
        int32_t hdr[3];
        if (!recv_all(hdr, sizeof hdr)) return 0;
        const int op = hdr[0], pos0 = hdr[1], n = hdr[2];
        if (op == 3) {
            if (nx != INVALID_SOCKET) { int32_t bye[3] = { 3, 0, 0 }; sendn(nx, bye, sizeof bye); }
            return 0;
        }
        int32_t status = 0;
        if (op == 1 || op == 2 || op == 5) {
            int32_t * ids = malloc((size_t) (n > 0 ? n : 1) * 4);
            if (n > 0 && !recv_all(ids, n * 4)) return 0;
            if (op == 5) {
                float * rows = malloc((size_t) n * n_embd * 4);
                steps(n, ids, NULL, pos0, rows);
                if (!pass_on(pos0, n, rows)) status = 1;
                send_all(&status, 4);
                if (!status) send_all(rows, n * n_embd * 4);
                free(rows);
            } else {
                steps(n, ids, NULL, pos0, out);
                send_all(&status, 4); send_all(out, n_vocab * 4);
            }
            free(ids);
        } else if (op == 4) {
            float * rows = malloc((size_t) n * n_embd * 4);
            if (!recv_all(rows, n * n_embd * 4)) return 0;
            float * done = malloc((size_t) n * n_embd * 4);
            steps(n, NULL, rows, pos0, done);
            memcpy(rows, done, (size_t) n * n_embd * 4);
            free(done);
            if (!pass_on(pos0, n, rows)) status = 1;
            send_all(&status, 4);
            if (!status) send_all(rows, n * n_embd * 4);
            free(rows);
        } else {
            status = 1;
            send_all(&status, 4);
        }
    }
}

// ------------------------------------------------------------------ main
int main(int argc, char ** argv) {
    const char * model = NULL;
    int pick = 0, port = 0, do_grade = 0, do_list = 0, ctx = 4096, l0 = 0, l1 = 0, threads = 0;
    const char * chain = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--list")) do_list = 1;
        else if (!strcmp(argv[i], "--grade")) do_grade = 1;
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) pick = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--connect") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) { sscanf(argv[++i], "%d:%d", &l0, &l1); }
        else if (!strcmp(argv[i], "--chain") && i + 1 < argc) chain = argv[++i];
        else if (argv[i][0] != '-') model = argv[i];
    }
    if (getenv("SML_VK_SKIP")) g_skip = atoi(getenv("SML_VK_SKIP"));
    if (!vk_load()) { fprintf(stderr, "no Vulkan driver\n"); return 2; }
    if (do_list) { list_devices(-1, 1); return 0; }
    if (!model) { fprintf(stderr, "usage: sml-vk --list | MODEL.ssf [--grade | --connect PORT] [--device N] [--layers A:B] [--ctx N]\n"); return 2; }
    if (threads <= 0) { SYSTEM_INFO si; GetSystemInfo(&si); threads = (int) si.dwNumberOfProcessors / 2; if (threads < 1) threads = 1; }
    char dir[1024];
    GetModuleFileNameA(NULL, dir, sizeof dir);
    char * slash = strrchr(dir, '\\');
    if (slash) *slash = 0;
    char why[512] = "";
    int ok = open_device(pick);
    if (!ok) snprintf(why, sizeof why, "no Vulkan device %d", pick);
    if (ok && !make_pipelines(dir)) { ok = 0; snprintf(why, sizeof why, "the Vulkan kernels (vk\\*.spv) are missing"); }
    const int head = l1 == 0 && l0 == 0;          // no --layers: the whole model and its head
    if (ok) {
        LARGE_INTEGER f, a, b; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&a);
        ok = load(model, ctx, threads, l0, l1, head || do_grade ? head : 0, why, sizeof why);
        QueryPerformanceCounter(&b);
        if (ok) fprintf(stderr, "  sml-vk: %s, layers %d..%d%s, loaded in %.1f s\n", dev_name, L0, L1 - 1,
                        want_head ? " + head" : "", (double) (b.QuadPart - a.QuadPart) / (double) f.QuadPart);
    }
    if (ok && chain && *chain && !want_head) ok = start_next(model, ctx, chain, why, sizeof why);
    if (port) return serve(port, ok, ok ? dev_name : why);
    if (!ok) { printf("  %s\n", why); return 1; }
    if (do_grade) return grade(threads);
    printf("  loaded; nothing asked\n");
    return 0;
}
