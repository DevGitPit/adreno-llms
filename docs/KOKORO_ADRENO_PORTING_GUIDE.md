# Kokoro-82M on Qualcomm Adreno GPU: Kernel Generation, Tuning & Deployment Guide

This guide details the complete architecture, OpenCL kernel compilation and caching lifecycle, hardware-specific tuning across Qualcomm Adreno GPU generations, voice pack extraction, and APK stripping for the on-device Kokoro-82M TTS engine.

---

## 1. Qualcomm Adreno GPU Architecture & OpenCL Execution

### GPU Architectures & Wavefront Characteristics

| GPU Generation | Microarchitecture | Representative SoCs | FP16 Compute | FP32 Generator Mode |
|---|---|---|---|---|
| **Adreno 6xx** (640, 650, 660) | A6xx (Vec4/Scalar) | Snapdragon 855, 865, 888 | Supported (`cl_khr_fp16`) | Recommended |
| **Adreno 7xx** (730, 732, 740, 750) | A7xx (Wave64 scalar ALUs) | Snapdragon 8 Gen 1, 7+ Gen 3, 8 Gen 2, 8 Gen 3 | Native fast half-precision | **Mandatory** (`NNOPT_GPU_FP32_GENERATOR=1`) |
| **Adreno 8xx** (830) | Oryon/Adreno 8xx | Snapdragon 8 Elite | Ultra-wide unified ALU | **Mandatory** (`NNOPT_GPU_FP32_GENERATOR=1`) |

### Critical Hardware Behaviors
1. **Unified Physical Memory (Zero-Copy):**
   - Adreno shares system LPDDR5/5X RAM with the CPU Kryo cores.
   - Memory buffers created with `CL_MEM_ALLOC_HOST_PTR` or mapped using `clEnqueueMapBuffer` do not require PCIe bus copies.
2. **Wavefront Size:**
   - Adreno typically executes work-groups in wavefronts (subgroups) of **64 work-items** (or 32/128 depending on register pressure).
   - Local work size should generally be multiples of 64 (e.g., 64, 128, 256) for optimal occupancy.
3. **Vendor Driver Paths:**
   - Stock system OpenCL driver: `/vendor/lib64/libOpenCL.so` or `/vendor/lib64/egl/libGLESv2_adreno.so`.
   - On rooted / Shizuku / Termux setups, dlopen typically succeeds against `/vendor/lib64/libOpenCL.so` or `/system/vendor/lib64/libOpenCL.so`.

---

## 2. Kernel Compilation, Caching & Generation

### Online JIT vs. Cached Binary Execution

Compiling all Kokoro OpenCL kernels (BERT attention, duration predictor, LSTM, HiFi-GAN residual blocks, ISTFT) from raw source strings on app startup takes **5 to 8 seconds** on mobile devices.

To eliminate this delay, the engine implements binary program caching:

```
[Kernel Source + Compiler Flags + Device Name Hash]
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
  [Cache Hit: .bin exists]  [Cache Miss / First Run]
         │                       │
  clCreateProgramWithBinary  clCreateProgramWithSource
         │                       │
      ~5-10ms              clBuildProgram (~5000ms)
                                 │
                            clGetProgramInfo(CL_PROGRAM_BINARIES)
                                 │
                            Write to $NNOPT_KERNEL_CACHE_DIR/*.bin
```

### The Caching Subsystem (`opencl_context.cpp`)
- Function: `nnopt_build_program_cached(ctx, dev, source, options, cache_key)`
- Cache Key Format: `<cache_key>.<source_options_device_hash>.bin`
- Default Location: `$NNOPT_KERNEL_CACHE_DIR` (set in `jni_bridge.cpp` to `<app_files_dir>/kokoro_gpu/kernel_cache`).
- **Cache Invalidation:** If the kernel source, compiler flags, or device name change, the hash changes, triggering an automatic recompile and updating the `.bin` cache.

### Important: Non-Transferability Across Different Devices
> [!WARNING]
> Qualcomm OpenCL binary blobs (`.bin`) contain compiled machine code specific to the exact GPU core and driver build.
> - An Adreno 732 binary **cannot** run on an Adreno 660 or Adreno 830.
> - When deploying to a different device, delete existing `.bin` files or allow the engine to regenerate them on first launch.

### Precision Enforcement: Why Generator Must Be FP32
- **BERT Encoder & Predictors:** Execute in OpenCL FP16 (`cl_khr_fp16`). This halves memory bandwidth and doubles ALU execution speed without degradation.
- **Generator (HiFi-GAN / Waveform Decoder):**
  - Uses recursive Conv1D, TransposeConv1D, and ISTFT synthesis.
  - In FP16, repeated cumulative operations exceed the 16-bit half-precision dynamic range ($[-65504, 65504]$) during audio waveform accumulation, producing audible harsh cracking and digital distortion.
  - Setting `export NNOPT_GPU_FP32_GENERATOR=1` in `init()` forces `_generator_fp32.cpp`, producing pristine 24kHz studio-quality PCM.

---

## 3. CLBlast GEMM Tuning for New GPUs

Kokoro utilizes `CLBlast` for matrix multiplications (`Xgemm`). While default heuristics work well, you can run the CLBlast offline tuner on a new device to extract maximum FLOPS:

```bash
# In Termux on the target device:
# 1. Build CLBlast tuning tools
cd /path/to/CLBlast/build
cmake -DTUNERS=ON .. && make -j$(nproc)

# 2. Run tuner for Half Precision (FP16)
./clblast_tune_xgemm_half --platform 0 --device 0 --runs 2 --all_fraction 0.05 > clblast_adreno_fp16.tsv

# 3. Run tuner for Single Precision (FP32)
./clblast_tune_xgemm_float --platform 0 --device 0 --runs 2 --all_fraction 0.05 > clblast_adreno_fp32.tsv
```

These parameters can be baked into CLBlast or placed in `/data/data/com.kokoros/files/` as a tuned database.

---

## 4. Voice Packs & Style Embedding Generation

Kokoro-82M voices are represented by a **`[510, 256]` float32 tensor** (510 sequence length indices $\times$ 256 style vector dimensions).

### Converting Voice `.pt` to Runtime `.bin`
Given a `.pt` voice checkpoint from [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices):

```bash
python scripts/convert_voice.py voices/am_adam.pt assets/voice_pack_am_adam.bin
```

Script logic:
```python
import torch
import numpy as np

# Load style embedding
v = torch.load("am_adam.pt", map_location="cpu", weights_only=True)
arr = v.detach().to(torch.float32).reshape(510, 256).contiguous().numpy()
assert arr.size == 510 * 256

# Write flat little-endian FP32 binary (522,240 bytes)
arr.astype("<f4").tofile("voice_pack_am_adam.bin")
```

### Adding Voices to the APK
1. **Standalone assets fallback:**
   Place `voice_pack_<voice_id>.bin` in `<app_dir>/kokoro_gpu/assets/voice_pack_<voice_id>.bin`.
2. **ZIP archive (`voices.bin`):**
   `voices.bin` in the app is an uncompressed `.zip` file containing `<voice_id>.npy` arrays:
   ```bash
   # Package .npy files without compression (STORED)
   zip -0 voices.bin *.npy
   ```
   The JNI loader directly memory-reads `voice + ".npy"` from the ZIP structure without unpacking.

---

## 5. Token Processing Architecture (510 Tokens vs Sub-Chunking)

### Sentence Prosody Retention
- Kokoro's position embedding table has 512 entries.
- The maximum supported input length is **510 tokens** (BOS + up to 508 phonemes + EOS).
- In [`jni_bridge.cpp`](file:///data/data/com.termux/files/home/adreno-llms/src/models/kokoro-82m/src/jni_bridge.cpp):
  - Inbound text is checked via `engine->phonemizer.phonemize(text_str)`.
  - If $\le 510$ tokens: executes `forward_graph` in **one monolithic pass**. Complete sentence pitch contour and prosody are preserved.
  - If $> 510$ tokens: splits at natural punctuation boundaries (`. `, `! `, `? `, `\n`) into groups of $\le 500$ tokens.

---

## 6. Stripping the APK of CPU-Related Bloat

In standard Kokoros APK builds, the package contains ONNX Runtime CPU libraries and ONNX model graphs:

| Component | Size | GPU Backend Necessity | Action |
|---|---|---|---|
| `libonnxruntime.so` | ~25.8 MB | **Unused** | Strip from `lib/arm64-v8a/` |
| `libpcaudio.so` | ~6.5 KB | Optional | Keep or replace with native ring-buffer |
| `model.onnx` / quantized models | ~80 - 160 MB | **Unused** | Remove from assets |
| `libkokoros_android.so` | ~670 KB | **Required** (OpenCL engine) | Keep |
| `libclblast.so` | ~6.5 MB | **Required** (OpenCL GEMM) | Keep |
| `libespeak-ng.so` + `espeak-ng-data.zip` | ~9.5 MB | **Required** (G2P phonemizer) | Keep |

### Steps for CPU Stripping:
1. Decompile APK:
   ```bash
   apktool d base_kokoros.apk -o apktool_out
   ```
2. Remove CPU native library:
   ```bash
   rm apktool_out/lib/arm64-v8a/libonnxruntime.so
   ```
3. Remove unused ONNX models:
   ```bash
   rm -f apktool_out/assets/*.onnx
   ```
4. **Bytecode Safety Rule:**
   Do **NOT** alter Smali method register counts or signatures in `classes*.dex`. If the Java layer has dead references to ONNX initialization that are never invoked at runtime, keep the smali signatures intact to avoid ART Dalvik `VerifyError`.

---

## 7. Step-by-Step Build & Deployment Workflow

### 1. Build Native Library in Termux
```bash
cd /path/to/adreno-llms/src/models/kokoro-82m/build_native
cmake --build . --target kokoros_android -j$(nproc)
patchelf --replace-needed libclblast.so.1 libclblast.so libkokoros_android.so
```

### 2. Stage Library into APK Workspace
```bash
cp libkokoros_android.so /path/to/apktool_out/lib/arm64-v8a/
```

### 3. Repackage, Align & Sign
```bash
cd /path/to/apk_workspace
apktool b apktool_out -o rebuilt_unaligned.apk
zipalign -p -f 4 rebuilt_unaligned.apk rebuilt_aligned.apk
apksigner sign --ks debug.keystore --ks-pass pass:android --key-pass pass:android --out kokoros_ready.apk rebuilt_aligned.apk
```

### 4. Install via Shizuku (`rish`)
```bash
rish -c "cp kokoros_ready.apk /data/local/tmp/k.apk && pm install -r -d /data/local/tmp/k.apk && rm /data/local/tmp/k.apk && am force-stop com.kokoros"
```

### 5. Inspect Runtime Logs
```bash
rish -c "logcat -d -s KokoroAdrenoJNI KokoroTTS AndroidRuntime"
```
Look for:
```text
KokoroAdrenoJNI: Generated <N> audio samples (<S>s) on Adreno GPU across 1 chunks
```
