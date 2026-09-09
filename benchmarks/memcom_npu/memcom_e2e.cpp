// memcom_e2e.cpp - true end-to-end decode loop for memcom W8A16 QNN context binary
// Tensors are copied verbatim from the context binary's own metadata (libQnnSystem.so)
// so ids/type/quantization match exactly; only buffers are ours. Then autoregressive
// decode: execute -> argmax -> KV/state feedback -> next token, with per-token timing.
#include <dlfcn.h>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

#include "QnnTypes.h"
#include "QnnInterface.h"
#include "QnnBackend.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnCommon.h"
#include "QnnDevice.h"
#include "System/QnnSystemInterface.h"
#include "System/QnnSystemContext.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif

struct TensorRT {
  std::string name;
  std::vector<uint32_t> dims;
  Qnn_DataType_t dt;
  float scale;
  int32_t offset;
  bool isInput;
  size_t nElems;
  std::vector<uint8_t> buf;
  Qnn_Tensor_t qt;
};

struct Pair {
  TensorRT* in;
  TensorRT* out;
};

static QnnHtpDevice_PerfInfrastructure_t* g_perfInfra = nullptr;
static uint32_t g_powerConfigId = 0;

static bool setupPerfVoting(QnnInterface_t& qnn, Qnn_BackendHandle_t backend) {
  Qnn_DeviceHandle_t device = nullptr;
  if (QNN_SUCCESS != qnn.QNN_INTERFACE_VER_NAME.deviceCreate(nullptr, nullptr, &device)) {
    fprintf(stderr, "vote: deviceCreate failed\n");
    return false;
  }
  QnnHtpDevice_Infrastructure_t htpInfra{};
  htpInfra.infraType = QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF;
  QnnDevice_Infrastructure_t infraPtr = (QnnDevice_Infrastructure_t)&htpInfra;
  if (QNN_SUCCESS != qnn.QNN_INTERFACE_VER_NAME.deviceGetInfrastructure(&infraPtr)) {
    fprintf(stderr, "vote: deviceGetInfrastructure failed\n");
    return false;
  }
  g_perfInfra = &htpInfra.perfInfra;
  if (QNN_SUCCESS != g_perfInfra->createPowerConfigId(0, 0, &g_powerConfigId)) {
    fprintf(stderr, "vote: createPowerConfigId failed\n");
    return false;
  }
  QnnHtpPerfInfrastructure_PowerConfig_t dcvs{};
  dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  dcvs.dcvsV3Config.contextId = 0;
  dcvs.dcvsV3Config.setDcvsEnable = 1;
  dcvs.dcvsV3Config.dcvsEnable = 1;
  dcvs.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  dcvs.dcvsV3Config.setSleepLatency = 0;
  dcvs.dcvsV3Config.sleepLatency = 0;
  dcvs.dcvsV3Config.setSleepDisable = 1;
  dcvs.dcvsV3Config.sleepDisable = 1;
  dcvs.dcvsV3Config.setBusParams = 1;
  dcvs.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO;
  dcvs.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;
  dcvs.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO;
  dcvs.dcvsV3Config.setCoreParams = 1;
  dcvs.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO;
  dcvs.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;
  dcvs.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO;

  QnnHtpPerfInfrastructure_PowerConfig_t rpc{};
  rpc.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
  rpc.rpcPollingTimeConfig = 9999;

  QnnHtpPerfInfrastructure_PowerConfig_t ddr{};
  ddr.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DDR_PERF_MODE;
  ddr.ddrPerfModeConfig = 1;

  const QnnHtpPerfInfrastructure_PowerConfig_t* configs[] = {&dcvs, &rpc, &ddr, nullptr};
  if (QNN_SUCCESS != g_perfInfra->setPowerConfig(g_powerConfigId, configs)) {
    fprintf(stderr, "vote: setPowerConfig failed (DDR_PERF_MODE may need TURBO bus corner)\n");
    return false;
  }
  printf("vote: DCVS TURBO + RPC polling 9999us + DDR perf mode ON\n");
  return true;
}

static void teardownPerfVoting() {
  if (g_perfInfra && g_powerConfigId) {
    g_perfInfra->destroyPowerConfigId(g_powerConfigId);
    g_powerConfigId = 0;
  }
}

static void neonConvert(const uint16_t* src, uint16_t* dst, size_t n, float A, float B) {
#ifdef __aarch64__
  float32x4_t va = vdupq_n_f32(A);
  float32x4_t vb = vdupq_n_f32(B);
  float32x4_t vhalf = vdupq_n_f32(0.5f);
  float32x4_t vmax = vdupq_n_f32(65535.0f);
  float32x4_t vmin = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    uint16x4_t vq = vld1_u16(&src[i]);
    uint32x4_t vu32 = vmovl_u16(vq);
    float32x4_t vf = vcvtq_f32_u32(vu32);
    vf = vmlaq_f32(vb, vf, va);
    vf = vaddq_f32(vf, vhalf);
    vf = vmaxq_f32(vf, vmin);
    vf = vminq_f32(vf, vmax);
    uint32x4_t vout = vcvtq_u32_f32(vf);
    vst1_u16(&dst[i], vmovn_u32(vout));
  }
  for (; i < n; i++) {
    float v = A * (float)src[i] + B;
    if (v < 0.0f) v = 0.0f;
    if (v > 65535.0f) v = 65535.0f;
    dst[i] = (uint16_t)v;
  }
#else
  for (size_t i = 0; i < n; i++) {
    float v = A * (float)src[i] + B;
    if (v < 0.0f) v = 0.0f;
    if (v > 65535.0f) v = 65535.0f;
    dst[i] = (uint16_t)v;
  }
#endif
}

// QNN convention: real = scale * (q + offset); q = round(real / scale) - offset
static inline uint16_t quant16(float x, float s, int32_t o) {
  float q = x / s - (float)o;
  if (q < 0.0f) q = 0.0f;
  if (q > 65535.0f) q = 65535.0f;
  return (uint16_t)(q + 0.5f);
}
static inline float dequant16(uint16_t q, float s, int32_t o) {
  return s * ((float)q + (float)o);
}

static bool readFile(const std::string& p, std::vector<uint8_t>& out) {
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  out.resize((size_t)n);
  size_t rd = fread(out.data(), 1, (size_t)n, f);
  fclose(f);
  return rd == (size_t)n;
}

int main(int argc, char** argv) {
  int tokens = 50, warmup = 3, prefill = 0, vote = 0;
  std::string ctxPath = "memcom_decode_w8a16.bin", inputsDir = "inputs";
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--tokens" && i + 1 < argc) tokens = atoi(argv[++i]);
    else if (a == "--warmup" && i + 1 < argc) warmup = atoi(argv[++i]);
    else if (a == "--prefill" && i + 1 < argc) prefill = atoi(argv[++i]);
    else if (a == "--vote") vote = 1;
    else if (a == "--context" && i + 1 < argc) ctxPath = argv[++i];
    else if (a == "--inputs" && i + 1 < argc) inputsDir = argv[++i];
  }
  if (prefill > 0) printf("prefill: %d tokens (zero-init state)\n", prefill);

  // 1. dlopen HTP backend
  void* h = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
  if (!h) {
    fprintf(stderr, "dlopen libQnnHtp.so failed: %s\n", dlerror());
    return 1;
  }
  typedef Qnn_ErrorHandle_t (*GetProvFn)(const QnnInterface_t***, uint32_t*);
  GetProvFn gp = (GetProvFn)dlsym(h, "QnnInterface_getProviders");
  if (!gp) {
    fprintf(stderr, "dlsym QnnInterface_getProviders failed: %s\n", dlerror());
    return 1;
  }
  const QnnInterface_t** provs = nullptr;
  uint32_t nprov = 0;
  if (QNN_SUCCESS != gp(&provs, &nprov) || nprov == 0) {
    fprintf(stderr, "getProviders failed\n");
    return 1;
  }
  QnnInterface_t qnn = *provs[0];

  Qnn_BackendHandle_t backend = nullptr;
  if (QNN_SUCCESS != qnn.QNN_INTERFACE_VER_NAME.backendCreate(nullptr, nullptr, &backend)) {
    fprintf(stderr, "backendCreate failed\n");
    return 1;
  }
  if (vote) {
    if (!setupPerfVoting(qnn, backend)) {
      fprintf(stderr, "vote: setup failed, continuing without perf voting\n");
    }
  }

  // 2. read serialized context
  std::vector<uint8_t> bin;
  if (!readFile(ctxPath, bin)) {
    fprintf(stderr, "failed to read %s\n", ctxPath.c_str());
    return 1;
  }
  printf("context binary: %.1f MB\n", bin.size() / 1e6);

  // 3. inspect binary metadata via system lib -> exact tensor structs
  void* hs = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
  if (!hs) {
    fprintf(stderr, "dlopen libQnnSystem.so failed: %s\n", dlerror());
    return 1;
  }
  typedef Qnn_ErrorHandle_t (*GetSysProvFn)(const QnnSystemInterface_t***, uint32_t*);
  GetSysProvFn gsp = (GetSysProvFn)dlsym(hs, "QnnSystemInterface_getProviders");
  if (!gsp) {
    fprintf(stderr, "dlsym QnnSystemInterface_getProviders failed: %s\n", dlerror());
    return 1;
  }
  const QnnSystemInterface_t** sprovs = nullptr;
  uint32_t nsprov = 0;
  if (QNN_SUCCESS != gsp(&sprovs, &nsprov) || nsprov == 0) {
    fprintf(stderr, "system getProviders failed\n");
    return 1;
  }
  QnnSystemInterface_t qsys = *sprovs[0];

  QnnSystemContext_Handle_t sysCtx = nullptr;
  if (QNN_SUCCESS != qsys.QNN_SYSTEM_INTERFACE_VER_NAME.systemContextCreate(&sysCtx)) {
    fprintf(stderr, "systemContextCreate failed\n");
    return 1;
  }
  const QnnSystemContext_BinaryInfo_t* bi = nullptr;
  Qnn_ContextBinarySize_t biSize = 0;
  if (QNN_SUCCESS != qsys.QNN_SYSTEM_INTERFACE_VER_NAME.systemContextGetBinaryInfo(
                           sysCtx, bin.data(), (uint64_t)bin.size(), &bi, &biSize)) {
    fprintf(stderr, "systemContextGetBinaryInfo failed\n");
    return 1;
  }

  const QnnSystemContext_GraphInfo_t* graphs = nullptr;
  uint32_t numGraphs = 0;
  switch (bi->version) {
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
      graphs = bi->contextBinaryInfoV1.graphs;
      numGraphs = bi->contextBinaryInfoV1.numGraphs;
      break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
      graphs = bi->contextBinaryInfoV2.graphs;
      numGraphs = bi->contextBinaryInfoV2.numGraphs;
      break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
      graphs = bi->contextBinaryInfoV3.graphs;
      numGraphs = bi->contextBinaryInfoV3.numGraphs;
      break;
    default:
      fprintf(stderr, "unsupported binary info version %u\n", (unsigned)bi->version);
      return 1;
  }
  if (numGraphs != 1) {
    fprintf(stderr, "expected 1 graph, got %u\n", numGraphs);
    return 1;
  }

  std::string graphName;
  const Qnn_Tensor_t* tins = nullptr;
  const Qnn_Tensor_t* touts = nullptr;
  uint32_t nIn = 0, nOut = 0;
  const QnnSystemContext_GraphInfo_t& ginfo = graphs[0];
  switch (ginfo.version) {
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1:
      graphName = ginfo.graphInfoV1.graphName;
      tins = ginfo.graphInfoV1.graphInputs;
      nIn = ginfo.graphInfoV1.numGraphInputs;
      touts = ginfo.graphInfoV1.graphOutputs;
      nOut = ginfo.graphInfoV1.numGraphOutputs;
      break;
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2:
      graphName = ginfo.graphInfoV2.graphName;
      tins = ginfo.graphInfoV2.graphInputs;
      nIn = ginfo.graphInfoV2.numGraphInputs;
      touts = ginfo.graphInfoV2.graphOutputs;
      nOut = ginfo.graphInfoV2.numGraphOutputs;
      break;
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3:
      graphName = ginfo.graphInfoV3.graphName;
      tins = ginfo.graphInfoV3.graphInputs;
      nIn = ginfo.graphInfoV3.numGraphInputs;
      touts = ginfo.graphInfoV3.graphOutputs;
      nOut = ginfo.graphInfoV3.numGraphOutputs;
      break;
    default:
      fprintf(stderr, "unsupported graph info version %u\n", (unsigned)ginfo.version);
      return 1;
  }
  printf("graph: %s  inputs: %u  outputs: %u\n", graphName.c_str(), nIn, nOut);

  // 4. context from binary + retrieve graph
  Qnn_ContextHandle_t context = nullptr;
  auto tb = std::chrono::steady_clock::now();
  if (QNN_SUCCESS != qnn.QNN_INTERFACE_VER_NAME.contextCreateFromBinary(
                         backend, nullptr, nullptr, bin.data(), bin.size(), &context, nullptr)) {
    fprintf(stderr, "contextCreateFromBinary failed\n");
    return 1;
  }
  auto td = std::chrono::steady_clock::now();
  printf("context load: %.1f ms\n", std::chrono::duration<double, std::milli>(td - tb).count());

  Qnn_GraphHandle_t graph = nullptr;
  if (QNN_SUCCESS != qnn.QNN_INTERFACE_VER_NAME.graphRetrieve(context, graphName.c_str(), &graph)) {
    fprintf(stderr, "graphRetrieve(%s) failed\n", graphName.c_str());
    return 1;
  }

  // 5. build tensor storage: copy exact Qnn_Tensor_t from metadata, own the buffers
  std::vector<TensorRT> ins, outs;
  ins.reserve(nIn);
  outs.reserve(nOut);
  auto addTensor = [&](const Qnn_Tensor_t& src, bool isInput) -> bool {
    TensorRT t{};
    t.qt = src;
    const char* nm;
    uint32_t rank;
    uint32_t* dimsArr;
    Qnn_DataType_t dt;
    if (src.version == QNN_TENSOR_VERSION_1) {
      nm = src.v1.name;
      rank = src.v1.rank;
      dimsArr = src.v1.dimensions;
      dt = src.v1.dataType;
      t.scale = src.v1.quantizeParams.scaleOffsetEncoding.scale;
      t.offset = src.v1.quantizeParams.scaleOffsetEncoding.offset;
    } else if (src.version == QNN_TENSOR_VERSION_2) {
      nm = src.v2.name;
      rank = src.v2.rank;
      dimsArr = src.v2.dimensions;
      dt = src.v2.dataType;
      t.scale = src.v2.quantizeParams.scaleOffsetEncoding.scale;
      t.offset = src.v2.quantizeParams.scaleOffsetEncoding.offset;
    } else {
      fprintf(stderr, "unexpected tensor version %u\n", (unsigned)src.version);
      return false;
    }
    t.name = nm;
    t.dt = dt;
    t.isInput = isInput;
    t.nElems = 1;
    for (uint32_t r = 0; r < rank; r++) {
      t.dims.push_back(dimsArr[r]);
      t.nElems *= dimsArr[r];
    }
    size_t elemSize;
    if (dt == QNN_DATATYPE_UFIXED_POINT_16) elemSize = 2;
    else if (dt == QNN_DATATYPE_INT_32) elemSize = 4;
    else {
      fprintf(stderr, "%s: unsupported dtype %u\n", nm, (unsigned)dt);
      return false;
    }
    t.buf.resize(t.nElems * elemSize, 0);
    if (isInput) ins.push_back(std::move(t));
    else outs.push_back(std::move(t));
    return true;
  };
  for (uint32_t i = 0; i < nIn; i++) {
    if (!addTensor(tins[i], true)) return 1;
  }
  for (uint32_t i = 0; i < nOut; i++) {
    if (!addTensor(touts[i], false)) return 1;
  }

  // 6. re-point name/dims/clientBuf to our stable storage (metadata freed later)
  std::map<std::string, TensorRT*> byName;
  auto fixup = [&](std::vector<TensorRT>& v) {
    for (TensorRT& t : v) {
      if (t.qt.version == QNN_TENSOR_VERSION_1) {
        t.qt.v1.name = t.name.c_str();
        t.qt.v1.dimensions = t.dims.data();
        t.qt.v1.memType = QNN_TENSORMEMTYPE_RAW;
        t.qt.v1.clientBuf.data = t.buf.data();
        t.qt.v1.clientBuf.dataSize = (uint32_t)t.buf.size();
      } else {
        t.qt.v2.name = t.name.c_str();
        t.qt.v2.dimensions = t.dims.data();
        t.qt.v2.memType = QNN_TENSORMEMTYPE_RAW;
        t.qt.v2.clientBuf.data = t.buf.data();
        t.qt.v2.clientBuf.dataSize = (uint32_t)t.buf.size();
      }
      byName[t.name] = &t;
    }
  };
  fixup(ins);
  fixup(outs);
  qsys.QNN_SYSTEM_INTERFACE_VER_NAME.systemContextFree(sysCtx);

  // 7. seed inputs from float32 raw files (ids.raw is int32)
  //    In prefill mode: zero-init state tensors (s/kc/vc), only seed ids/cos/sin
  for (TensorRT& t : ins) {
    bool isState = (t.name.size() > 1 && t.name[0] == 's' && isdigit((unsigned char)t.name[1]))
                   || t.name.rfind("kc", 0) == 0 || t.name.rfind("vc", 0) == 0;
    if (prefill > 0 && isState) {
      memset(t.buf.data(), 0, t.buf.size());
      continue;
    }
    std::vector<uint8_t> raw;
    if (!readFile(inputsDir + "/" + t.name + ".raw", raw)) {
      fprintf(stderr, "missing %s/%s.raw\n", inputsDir.c_str(), t.name.c_str());
      return 1;
    }
    if (t.name == "ids") {
      if (raw.size() < 4) {
        fprintf(stderr, "ids.raw too small\n");
        return 1;
      }
      memcpy(t.buf.data(), raw.data(), 4);
      continue;
    }
    size_t n = raw.size() / 4;
    if (n != t.nElems) {
      fprintf(stderr, "%s.raw elems %zu != tensor %zu\n", t.name.c_str(), n, t.nElems);
      return 1;
    }
    const float* f = (const float*)raw.data();
    uint16_t* q = (uint16_t*)t.buf.data();
    for (size_t i = 0; i < n; i++) q[i] = quant16(f[i], t.scale, t.offset);
  }

  // 8. input<->output feedback pairs: s{i}<-s_out{i}, kc{i}<-kc_out{i}, vc{i}<-vc_out{i}
  std::vector<Pair> pairs;
  int nConv = 0;
  for (TensorRT& in : ins) {
    std::string oname;
    if (in.name.size() > 1 && in.name[0] == 's' && isdigit((unsigned char)in.name[1]))
      oname = "s_out" + in.name.substr(1);
    else if (in.name.rfind("kc", 0) == 0) oname = "kc_out" + in.name.substr(2);
    else if (in.name.rfind("vc", 0) == 0) oname = "vc_out" + in.name.substr(2);
    else continue;
    auto it = byName.find(oname);
    if (it == byName.end()) {
      fprintf(stderr, "no output tensor for %s\n", in.name.c_str());
      return 1;
    }
    if (!(in.scale == it->second->scale && in.offset == it->second->offset)) nConv++;
    pairs.push_back({&in, it->second});
  }
  printf("feedback pairs: %zu (%d need quant conversion)\n", pairs.size(), nConv);

  TensorRT* logitsOut = byName["logits"];
  TensorRT* idsIn = byName["ids"];
  if (!logitsOut || !idsIn) {
    fprintf(stderr, "logits/ids tensor not found\n");
    return 1;
  }
  int32_t* idsPtr = (int32_t*)idsIn->buf.data();
  const uint16_t* lq = (const uint16_t*)logitsOut->buf.data();

  std::vector<Qnn_Tensor_t> qinsArr, qoutsArr;
  qinsArr.reserve(ins.size());
  qoutsArr.reserve(outs.size());
  for (TensorRT& t : ins) qinsArr.push_back(t.qt);
  for (TensorRT& t : outs) qoutsArr.push_back(t.qt);

  // 9. main loop: prefill + warmup + decode
  std::vector<double> prefillTimes, decodeTimes;
  int total = prefill + warmup + tokens;
  for (int step = 0; step < total; step++) {
    auto t0 = std::chrono::steady_clock::now();
    Qnn_ErrorHandle_t es = qnn.QNN_INTERFACE_VER_NAME.graphExecute(
        graph, qinsArr.data(), (uint32_t)qinsArr.size(), qoutsArr.data(), (uint32_t)qoutsArr.size(),
        nullptr, nullptr);
    if (QNN_SUCCESS != es) {
      fprintf(stderr, "graphExecute failed step %d: %u\n", step, (unsigned)es);
      return 1;
    }
    uint16_t best = 0;
    int bestIdx = 0;
    for (size_t i = 0; i < logitsOut->nElems; i++) {
      if (lq[i] > best) {
        best = lq[i];
        bestIdx = (int)i;
      }
    }
    *idsPtr = bestIdx;
    for (Pair& p : pairs) {
      if (p.in->scale == p.out->scale && p.in->offset == p.out->offset) {
        memcpy(p.in->buf.data(), p.out->buf.data(), p.in->buf.size());
      } else {
        float A = p.out->scale / p.in->scale;
        float B = p.out->scale * (float)p.out->offset / p.in->scale - (float)p.in->offset;
        neonConvert((const uint16_t*)p.out->buf.data(), (uint16_t*)p.in->buf.data(),
                    p.in->nElems, A, B);
      }
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (step < prefill) {
      prefillTimes.push_back(ms);
      if (step % 10 == 0 || step == prefill - 1)
        printf("prefill %3d/%d: %8.2f ms\n", step + 1, prefill, ms);
    } else if (step >= prefill + warmup) {
      decodeTimes.push_back(ms);
      printf("tok %2d: %8.2f ms  id=%5d  logit=%.3f\n", step - prefill - warmup, ms, bestIdx,
             dequant16(best, logitsOut->scale, logitsOut->offset));
    }
  }

  // 10. stats
  if (prefill > 0 && !prefillTimes.empty()) {
    double pTotal = 0, pMin = prefillTimes[0], pMax = prefillTimes[0];
    for (double v : prefillTimes) { pTotal += v; pMin = std::min(pMin, v); pMax = std::max(pMax, v); }
    double pAvg = pTotal / (double)prefillTimes.size();
    printf("\n=== PREFILL: %d tokens (zero-init state) ===\n", prefill);
    printf("total %.1f ms  min %.2f ms  avg %.2f ms  max %.2f ms\n", pTotal, pMin, pAvg, pMax);
    printf("throughput: %.2f tok/s (%.2f tok/s excl. first step)\n",
           1000.0 * prefill / pTotal,
           prefill > 1 ? 1000.0 * (prefill - 1) / (pTotal - prefillTimes[0]) : 0.0);
  }
  if (!decodeTimes.empty()) {
    std::sort(decodeTimes.begin(), decodeTimes.end());
    double mn = decodeTimes.front(), mx = decodeTimes.back(), avg = 0.0;
    for (double v : decodeTimes) avg += v;
    avg /= (double)decodeTimes.size();
    double p90 = decodeTimes[(size_t)((double)decodeTimes.size() * 0.9)];
    printf("\n=== DECODE: %d tokens (+%d warmup) ===\n", tokens, warmup);
    printf("min %.2f ms  avg %.2f ms  p90 %.2f ms  max %.2f ms\n", mn, avg, p90, mx);
    printf("throughput: %.2f tok/s (avg)\n", 1000.0 / avg);
  }

  teardownPerfVoting();
  qnn.QNN_INTERFACE_VER_NAME.contextFree(context, nullptr);
  qnn.QNN_INTERFACE_VER_NAME.backendFree(backend);
  dlclose(hs);
  dlclose(h);
  return 0;
}
