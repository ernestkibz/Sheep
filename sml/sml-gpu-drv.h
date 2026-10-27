// CUDA driver API, declared by hand and bound at runtime from nvcuda.dll.
//
// The point of doing it this way: nvcuda.dll ships with the DISPLAY DRIVER, not
// with the CUDA toolkit. This machine has the driver but no toolkit -- no nvcc,
// no import library, no headers. Binding the dozen entry points we actually use
// through GetProcAddress removes the toolkit from the build entirely.
#ifndef SML_DRV_H
#define SML_DRV_H

#include <stdio.h>
#include <windows.h>

typedef unsigned long long CUdeviceptr;
typedef int  CUdevice;
typedef void * CUcontext;
typedef void * CUmodule;
typedef void * CUfunction;
typedef void * CUstream;
typedef void * CUevent;
typedef void * CUgraph;
typedef void * CUgraphExec;

#define CU_OK 0

// Device attributes we query. Values are from the driver ABI and are stable.
#define CU_ATTR_MP_COUNT            16
#define CU_ATTR_CLOCK_RATE          13
#define CU_ATTR_MEM_CLOCK_RATE      36
#define CU_ATTR_MEM_BUS_WIDTH       37
#define CU_ATTR_CC_MAJOR            75
#define CU_ATTR_CC_MINOR            76

typedef int (__stdcall * pfn_cuInit)(unsigned);
typedef int (__stdcall * pfn_cuDeviceGet)(CUdevice *, int);
typedef int (__stdcall * pfn_cuDeviceGetName)(char *, int, CUdevice);
typedef int (__stdcall * pfn_cuDeviceGetAttribute)(int *, int, CUdevice);
typedef int (__stdcall * pfn_cuDeviceTotalMem)(size_t *, CUdevice);
typedef int (__stdcall * pfn_cuCtxCreate)(CUcontext *, unsigned, CUdevice);
typedef int (__stdcall * pfn_cuCtxDestroy)(CUcontext);
typedef int (__stdcall * pfn_cuCtxSynchronize)(void);
typedef int (__stdcall * pfn_cuModuleLoadData)(CUmodule *, const void *);
typedef int (__stdcall * pfn_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
typedef int (__stdcall * pfn_cuMemAlloc)(CUdeviceptr *, size_t);
typedef int (__stdcall * pfn_cuMemFree)(CUdeviceptr);
typedef int (__stdcall * pfn_cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
typedef int (__stdcall * pfn_cuMemcpyDtoH)(void *, CUdeviceptr, size_t);
typedef int (__stdcall * pfn_cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                                             unsigned, unsigned, unsigned,
                                             unsigned, CUstream, void **, void **);
typedef int (__stdcall * pfn_cuGetErrorString)(int, const char **);
// Streams and graphs: optional -- a driver without them keeps the per-token path.
typedef int (__stdcall * pfn_cuStreamCreate)(CUstream *, unsigned);
typedef int (__stdcall * pfn_cuStreamSynchronize)(CUstream);
typedef int (__stdcall * pfn_cuStreamBeginCapture)(CUstream, int);
typedef int (__stdcall * pfn_cuStreamEndCapture)(CUstream, CUgraph *);
typedef int (__stdcall * pfn_cuGraphInstantiateWithFlags)(CUgraphExec *, CUgraph, unsigned long long);
typedef int (__stdcall * pfn_cuGraphLaunch)(CUgraphExec, CUstream);
typedef int (__stdcall * pfn_cuGraphExecDestroy)(CUgraphExec);
typedef int (__stdcall * pfn_cuGraphDestroy)(CUgraph);
typedef int (__stdcall * pfn_cuFuncSetAttribute)(CUfunction, int, int);
typedef int (__stdcall * pfn_cuFuncGetAttribute)(int *, int, CUfunction);
typedef int (__stdcall * pfn_cuEventCreate)(CUevent *, unsigned);
typedef int (__stdcall * pfn_cuEventRecord)(CUevent, CUstream);
typedef int (__stdcall * pfn_cuStreamWaitEvent)(CUstream, CUevent, unsigned);
typedef int (__stdcall * pfn_cuModuleGetGlobal)(CUdeviceptr *, size_t *, CUmodule, const char *);
typedef int (__stdcall * pfn_cuMemAllocHost)(void **, size_t);
typedef int (__stdcall * pfn_cuMemFreeHost)(void *);
typedef int (__stdcall * pfn_cuMemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream);
typedef int (__stdcall * pfn_cuEventQuery)(CUevent);
typedef int (__stdcall * pfn_cuMemGetInfo)(size_t *, size_t *);

typedef struct {
    HMODULE lib;
    pfn_cuInit               Init;
    pfn_cuDeviceGet          DeviceGet;
    pfn_cuDeviceGetName      DeviceGetName;
    pfn_cuDeviceGetAttribute DeviceGetAttribute;
    pfn_cuDeviceTotalMem     DeviceTotalMem;
    pfn_cuCtxCreate          CtxCreate;
    pfn_cuCtxDestroy         CtxDestroy;
    pfn_cuCtxSynchronize     CtxSynchronize;
    pfn_cuModuleLoadData     ModuleLoadData;
    pfn_cuModuleGetFunction  ModuleGetFunction;
    pfn_cuMemAlloc           MemAlloc;
    pfn_cuMemFree            MemFree;
    pfn_cuMemcpyHtoD         MemcpyHtoD;
    pfn_cuMemcpyDtoH         MemcpyDtoH;
    pfn_cuLaunchKernel       LaunchKernel;
    pfn_cuGetErrorString     GetErrorString;
    pfn_cuStreamCreate       StreamCreate;
    pfn_cuStreamSynchronize  StreamSynchronize;
    pfn_cuStreamBeginCapture StreamBeginCapture;
    pfn_cuStreamEndCapture   StreamEndCapture;
    pfn_cuGraphInstantiateWithFlags GraphInstantiateWithFlags;
    pfn_cuGraphLaunch        GraphLaunch;
    pfn_cuGraphExecDestroy   GraphExecDestroy;
    pfn_cuGraphDestroy       GraphDestroy;
    pfn_cuFuncSetAttribute   FuncSetAttribute;
    pfn_cuFuncGetAttribute   FuncGetAttribute;
    pfn_cuEventCreate        EventCreate;
    pfn_cuEventRecord        EventRecord;
    pfn_cuStreamWaitEvent    StreamWaitEvent;
    pfn_cuModuleGetGlobal    ModuleGetGlobal;
    pfn_cuMemAllocHost       MemAllocHost;
    pfn_cuMemFreeHost        MemFreeHost;
    pfn_cuMemcpyHtoDAsync    MemcpyHtoDAsync;
    pfn_cuEventQuery         EventQuery;
    pfn_cuMemGetInfo         MemGetInfo;
} cu_api;

static cu_api CU;

#define CU_BIND(field, name)                                                   \
    do {                                                                       \
        CU.field = (pfn_cu##field) (void *) GetProcAddress(CU.lib, name);      \
        if (!CU.field) { printf("  missing entry point: %s\n", name); return 0; } \
    } while (0)

#define CU_BIND_OPT(field, name)                                               \
    CU.field = (pfn_cu##field) (void *) GetProcAddress(CU.lib, name)

static int cu_open(void) {
    CU.lib = LoadLibraryA("nvcuda.dll");
    if (!CU.lib) { printf("  nvcuda.dll not present (no NVIDIA driver)\n"); return 0; }
    // The _v2 suffixes are the current ABI for these calls; the unsuffixed
    // names are the pre-CUDA-4 versions and take different arguments.
    CU_BIND(Init,               "cuInit");
    CU_BIND(DeviceGet,          "cuDeviceGet");
    CU_BIND(DeviceGetName,      "cuDeviceGetName");
    CU_BIND(DeviceGetAttribute, "cuDeviceGetAttribute");
    CU_BIND(DeviceTotalMem,     "cuDeviceTotalMem_v2");
    CU_BIND(CtxCreate,          "cuCtxCreate_v2");
    CU_BIND(CtxDestroy,         "cuCtxDestroy_v2");
    CU_BIND(CtxSynchronize,     "cuCtxSynchronize");
    CU_BIND(ModuleLoadData,     "cuModuleLoadData");
    CU_BIND(ModuleGetFunction,  "cuModuleGetFunction");
    CU_BIND(MemAlloc,           "cuMemAlloc_v2");
    CU_BIND(MemFree,            "cuMemFree_v2");
    CU_BIND(MemcpyHtoD,         "cuMemcpyHtoD_v2");
    CU_BIND(MemcpyDtoH,         "cuMemcpyDtoH_v2");
    CU_BIND(LaunchKernel,       "cuLaunchKernel");
    CU_BIND(GetErrorString,     "cuGetErrorString");
    CU_BIND_OPT(StreamCreate,       "cuStreamCreate");
    CU_BIND_OPT(StreamSynchronize,  "cuStreamSynchronize");
    CU_BIND_OPT(StreamBeginCapture, "cuStreamBeginCapture_v2");
    CU_BIND_OPT(StreamEndCapture,   "cuStreamEndCapture");
    CU_BIND_OPT(GraphInstantiateWithFlags, "cuGraphInstantiateWithFlags");
    CU_BIND_OPT(GraphLaunch,        "cuGraphLaunch");
    CU_BIND_OPT(GraphExecDestroy,   "cuGraphExecDestroy");
    CU_BIND_OPT(GraphDestroy,       "cuGraphDestroy");
    CU_BIND_OPT(FuncSetAttribute,   "cuFuncSetAttribute");
    CU_BIND_OPT(FuncGetAttribute,   "cuFuncGetAttribute");
    CU_BIND_OPT(EventCreate,        "cuEventCreate");
    CU_BIND_OPT(EventRecord,        "cuEventRecord");
    CU_BIND_OPT(StreamWaitEvent,    "cuStreamWaitEvent");
    CU_BIND_OPT(ModuleGetGlobal,    "cuModuleGetGlobal_v2");
    CU_BIND_OPT(MemAllocHost,       "cuMemAllocHost_v2");
    CU_BIND_OPT(MemFreeHost,        "cuMemFreeHost");
    CU_BIND_OPT(MemcpyHtoDAsync,    "cuMemcpyHtoDAsync_v2");
    CU_BIND_OPT(EventQuery,         "cuEventQuery");
    CU_BIND_OPT(MemGetInfo,         "cuMemGetInfo_v2");
    return 1;
}

// NVML: the card's PHYSICAL memory, as nvidia-smi reports it. Under Windows
// (WDDM) cuMemGetInfo answers with the OS's budget for this process, which on
// this 4 GB card counted ~950 MB as taken where ~320 MB was (2026-09-26); a
// keep budget from it kept FEWER int8 copies than the old fixed guess.
typedef struct { unsigned long long total, free, used; } nvml_mem;
typedef int (__stdcall * pfn_nvmlInit)(void);
typedef int (__stdcall * pfn_nvmlDeviceGetHandleByPciBusId)(const char *, void **);
typedef int (__stdcall * pfn_nvmlDeviceGetHandleByIndex)(unsigned, void **);
typedef int (__stdcall * pfn_nvmlDeviceGetMemoryInfo)(void *, nvml_mem *);
typedef int (__stdcall * pfn_cuDeviceGetPCIBusId)(char *, int, CUdevice);

// Physical free bytes on device `dev`, or 0 when NVML cannot say.
static size_t nvml_free_bytes(CUdevice dev) {
    static HMODULE lib = NULL;
    static pfn_nvmlDeviceGetHandleByPciBusId by_bus = NULL;
    static pfn_nvmlDeviceGetHandleByIndex by_index = NULL;
    static pfn_nvmlDeviceGetMemoryInfo mem = NULL;
    static int tried = 0;
    if (!tried) {
        tried = 1;
        lib = LoadLibraryA("nvml.dll");
        if (!lib) lib = LoadLibraryA("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
        if (lib) {
            pfn_nvmlInit init = (pfn_nvmlInit) (void *) GetProcAddress(lib, "nvmlInit_v2");
            by_bus = (pfn_nvmlDeviceGetHandleByPciBusId) (void *) GetProcAddress(lib, "nvmlDeviceGetHandleByPciBusId_v2");
            by_index = (pfn_nvmlDeviceGetHandleByIndex) (void *) GetProcAddress(lib, "nvmlDeviceGetHandleByIndex_v2");
            mem = (pfn_nvmlDeviceGetMemoryInfo) (void *) GetProcAddress(lib, "nvmlDeviceGetMemoryInfo");
            if (!init || init() != 0) mem = NULL;
        }
    }
    if (!mem) return 0;
    void * h = NULL;
    // the same card CUDA gave us, found by PCI bus id; index 0 only as a fallback
    pfn_cuDeviceGetPCIBusId bus = CU.lib ? (pfn_cuDeviceGetPCIBusId) (void *) GetProcAddress(CU.lib, "cuDeviceGetPCIBusId") : NULL;
    char id[64];
    if (bus && by_bus && bus(id, sizeof id, dev) == CU_OK && by_bus(id, &h) == 0) {
        /* found */
    } else if (!by_index || by_index(0, &h) != 0) {
        return 0;
    }
    nvml_mem m;
    if (mem(h, &m) != 0) return 0;
    return (size_t) m.free;
}

static int cu_check(int rc, const char * what) {
    if (rc == CU_OK) return 1;
    const char * msg = "?";
    if (CU.GetErrorString) CU.GetErrorString(rc, &msg);
    printf("  %s failed: %d (%s)\n", what, rc, msg ? msg : "?");
    return 0;
}

#endif
