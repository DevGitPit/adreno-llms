#include <jni.h>
#include <android/log.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <mutex>
#include <map>
#include <algorithm>
#include <unistd.h>

#include "model.h"
#include "model_config.h"
#include "opencl_context.h"
#include "weights.h"
#include "phonemizer.h"
#include "load_bin.h"

#define LOG_TAG "KokoroAdrenoJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

struct KokoroEngineInstance {
    OpenCLContext cl_ctx;
    Weights weights;
    std::unique_ptr<Model> model;
    nnopt_tts::Phonemizer phonemizer;
    std::string voices_bin_path;
    std::string base_dir;
    std::mutex mtx;
    std::map<std::string, std::vector<float>> voice_cache;
    bool initialized = false;
};

// Helper: read uncompressed NPY float tensor from voices.bin (ZIP format)
static bool read_voice_from_npz(const std::string& zip_path, const std::string& voice, std::vector<float>& out_floats) {
    std::ifstream f(zip_path, std::ios::binary);
    if (!f.is_open()) {
        LOGE("Cannot open voices zip file: %s", zip_path.c_str());
        return false;
    }
    std::string target = voice + ".npy";
    while (f.good()) {
        uint32_t sig = 0;
        f.read(reinterpret_cast<char*>(&sig), 4);
        if (sig != 0x04034b50) break; // End of local files
        f.seekg(14, std::ios::cur);
        uint32_t comp_sz = 0, uncomp_sz = 0;
        f.read(reinterpret_cast<char*>(&comp_sz), 4);
        f.read(reinterpret_cast<char*>(&uncomp_sz), 4);
        uint16_t name_len = 0, extra_len = 0;
        f.read(reinterpret_cast<char*>(&name_len), 2);
        f.read(reinterpret_cast<char*>(&extra_len), 2);
        std::string fname(name_len, '\0');
        f.read(&fname[0], name_len);
        f.seekg(extra_len, std::ios::cur);
        if (fname == target) {
            // Read NPY header dynamically
            char magic_ver[8];
            f.read(magic_ver, 8);
            uint32_t hlen = 0;
            if (magic_ver[6] == 1) {
                uint16_t hlen16 = 0;
                f.read(reinterpret_cast<char*>(&hlen16), 2);
                hlen = hlen16;
            } else {
                f.read(reinterpret_cast<char*>(&hlen), 4);
            }
            f.seekg(hlen, std::ios::cur);
            out_floats.resize(510 * 256);
            f.read(reinterpret_cast<char*>(out_floats.data()), 510 * 256 * sizeof(float));
            return true;
        } else {
            f.seekg(comp_sz, std::ios::cur);
        }
    }
    return false;
}

// Splits text into natural sentence chunks where each chunk's phonemized token count is <= max_tokens (<= 510).
// eBook readers send text sentence by sentence (typically 10-40 words, ~20-60 tokens), so sentences run in a single
// pass up to 510 tokens, preserving natural prosody without artificial clause fragmentation.
static std::vector<std::string> chunk_text_by_tokens(
    const std::string& text,
    const nnopt_tts::Phonemizer& phonemizer,
    size_t max_tokens = 500
) {
    std::vector<int32_t> initial_ids = phonemizer.phonemize(text);
    if (initial_ids.size() <= max_tokens) {
        return { text };
    }

    // Split text on sentence boundaries (. ! ? \n)
    std::vector<std::string> sentences;
    std::string cur;
    for (size_t i = 0; i < text.size(); ++i) {
        cur += text[i];
        if (text[i] == '.' || text[i] == '!' || text[i] == '?' || text[i] == '\n') {
            if (i + 1 == text.size() || std::isspace(static_cast<unsigned char>(text[i + 1]))) {
                size_t b = cur.find_first_not_of(" \t\r\n");
                size_t e = cur.find_last_not_of(" \t\r\n");
                if (b != std::string::npos) {
                    sentences.push_back(cur.substr(b, e - b + 1));
                }
                cur.clear();
            }
        }
    }
    size_t b = cur.find_first_not_of(" \t\r\n");
    size_t e = cur.find_last_not_of(" \t\r\n");
    if (b != std::string::npos) {
        sentences.push_back(cur.substr(b, e - b + 1));
    }
    if (sentences.empty()) {
        sentences.push_back(text);
    }

    // Pack sentences into chunks <= max_tokens
    std::vector<std::string> chunks;
    std::string current_chunk;

    for (const auto& s : sentences) {
        std::string cand = current_chunk.empty() ? s : (current_chunk + " " + s);
        std::vector<int32_t> cand_ids = phonemizer.phonemize(cand);
        if (cand_ids.size() <= max_tokens) {
            current_chunk = cand;
        } else {
            if (!current_chunk.empty()) {
                chunks.push_back(current_chunk);
                current_chunk.clear();
            }
            std::vector<int32_t> s_ids = phonemizer.phonemize(s);
            if (s_ids.size() <= max_tokens) {
                current_chunk = s;
            } else {
                // If a single sentence exceeds max_tokens, fallback to clause chunking
                std::vector<std::string> sub = nnopt_tts::chunk_text(s, /*first_max_words=*/20, /*max_words=*/35);
                for (const auto& sc : sub) {
                    chunks.push_back(sc);
                }
            }
        }
    }
    if (!current_chunk.empty()) {
        chunks.push_back(current_chunk);
    }
    return chunks;
}

extern "C" {

JNIEXPORT jlong JNICALL Java_com_kokoros_KokoroJNI_init(
    JNIEnv* env,
    jclass clazz,
    jstring model_path,
    jstring voices_path,
    jstring espeak_data_path,
    jint intra_threads
) {
    LOGI("=== Initializing Kokoro Adreno GPU Engine ===");

    const char* c_model_path = env->GetStringUTFChars(model_path, nullptr);
    const char* c_voices_path = env->GetStringUTFChars(voices_path, nullptr);
    const char* c_espeak_path = env->GetStringUTFChars(espeak_data_path, nullptr);

    std::string m_path = c_model_path ? c_model_path : "";
    std::string v_path = c_voices_path ? c_voices_path : "";
    std::string e_path = c_espeak_path ? c_espeak_path : "";

    env->ReleaseStringUTFChars(model_path, c_model_path);
    env->ReleaseStringUTFChars(voices_path, c_voices_path);
    env->ReleaseStringUTFChars(espeak_data_path, c_espeak_path);

    // Determine base directory from model or espeak path
    std::string base_dir = e_path;
    // Set working dir so relative assets/weights paths resolve cleanly
    if (!base_dir.empty()) {
        chdir(base_dir.c_str());
        LOGI("Working directory set to: %s", base_dir.c_str());
    }

    auto engine = std::make_unique<KokoroEngineInstance>();
    engine->voices_bin_path = v_path;
    engine->base_dir = base_dir;

    // Initialize OpenCL context
    if (!engine->cl_ctx.initialize()) {
        LOGE("OpenCLContext initialization failed on Adreno GPU!");
        return 0;
    }

    // Set kernel cache directory inside app files
    std::string cache_dir = base_dir + "/kokoro_gpu/kernel_cache";
    setenv("NNOPT_KERNEL_CACHE_DIR", cache_dir.c_str(), 1);
    setenv("NNOPT_GPU_FP32_GENERATOR", "1", 1);
    setenv("NNOPT_DEBUG_LAYERS", "0", 1);

    // Locate weights
    std::string weights_bin = base_dir + "/kokoro_gpu/weights/model.fp16.bin";
    std::string weights_meta = base_dir + "/kokoro_gpu/weights/model.fp16.meta.json";

    // Fallback: check relative or parent
    if (access(weights_bin.c_str(), R_OK) != 0) {
        weights_bin = "weights/model.fp16.bin";
        weights_meta = "weights/model.fp16.meta.json";
    }

    LOGI("Loading weights from: %s", weights_bin.c_str());
    if (!engine->weights.load(weights_bin, weights_meta, engine->cl_ctx.context())) {
        LOGE("Weights::load() failed: %s", weights_bin.c_str());
        return 0;
    }

    // Pre-allocate GPU buffers for zero-allocation decode
    engine->weights.pre_allocate_all();

    // Initialize Model
    engine->model = std::make_unique<Model>(engine->cl_ctx, engine->weights);
    if (!engine->model->initialize()) {
        LOGE("Model::initialize() failed!");
        return 0;
    }

    // Initialize Phonemizer
    std::string vocab_tsv = base_dir + "/kokoro_gpu/assets/phoneme_vocab.tsv";
    if (access(vocab_tsv.c_str(), R_OK) != 0) {
        vocab_tsv = "assets/phoneme_vocab.tsv";
    }
    LOGI("Initializing Phonemizer with espeak parent: %s, vocab: %s", e_path.c_str(), vocab_tsv.c_str());
    if (!engine->phonemizer.init(e_path, vocab_tsv, "en-us")) {
        LOGW("Phonemizer::init returned false - will attempt fallback during synthesis if needed");
    }

    engine->initialized = true;
    LOGI("Kokoro Adreno GPU Engine successfully initialized! Ready for speech.");
    return reinterpret_cast<jlong>(engine.release());
}

JNIEXPORT jfloatArray JNICALL Java_com_kokoros_KokoroJNI_speak_1raw(
    JNIEnv* env,
    jclass clazz,
    jlong engine_ptr,
    jstring text,
    jstring voice,
    jstring language,
    jfloat speed
) {
    if (engine_ptr == 0) {
        LOGE("speak_raw called with NULL engine pointer");
        return nullptr;
    }

    auto* engine = reinterpret_cast<KokoroEngineInstance*>(engine_ptr);
    std::lock_guard<std::mutex> lock(engine->mtx);

    const char* c_text = env->GetStringUTFChars(text, nullptr);
    const char* c_voice = env->GetStringUTFChars(voice, nullptr);
    const char* c_lang = env->GetStringUTFChars(language, nullptr);

    std::string text_str = c_text ? c_text : "";
    std::string voice_str = c_voice ? c_voice : "af_heart";
    std::string lang_str = c_lang ? c_lang : "en-us";

    env->ReleaseStringUTFChars(text, c_text);
    env->ReleaseStringUTFChars(voice, c_voice);
    env->ReleaseStringUTFChars(language, c_lang);

    if (text_str.empty()) {
        return env->NewFloatArray(0);
    }

    // Ensure audio-correct generator is active
    setenv("NNOPT_GPU_FP32_GENERATOR", "1", 1);

    // Load voice tensor
    if (engine->voice_cache.find(voice_str) == engine->voice_cache.end()) {
        std::vector<float> vdata;
        bool loaded = false;
        if (!engine->voices_bin_path.empty()) {
            loaded = read_voice_from_npz(engine->voices_bin_path, voice_str, vdata);
        }
        if (!loaded) {
            // Check assets fallback
            std::string fallback_voice = engine->base_dir + "/kokoro_gpu/assets/voice_pack_" + voice_str + ".bin";
            if (access(fallback_voice.c_str(), R_OK) == 0) {
                vdata = load_float_bin(fallback_voice);
                loaded = (vdata.size() == 510 * 256);
            }
        }
        if (!loaded) {
            LOGW("Could not load voice %s, trying default af_heart", voice_str.c_str());
            std::string def_voice = engine->base_dir + "/kokoro_gpu/assets/voice_pack_af_heart.bin";
            vdata = load_float_bin(def_voice);
        }
        if (vdata.size() == 510 * 256) {
            engine->voice_cache[voice_str] = std::move(vdata);
        } else {
            LOGE("Failed to find any style vector for voice: %s", voice_str.c_str());
            return nullptr;
        }
    }

    const auto& voice_pack = engine->voice_cache[voice_str];

    std::vector<std::string> chunks = chunk_text_by_tokens(text_str, engine->phonemizer, /*max_tokens=*/500);
    if (chunks.empty()) chunks.push_back(text_str);

    std::vector<int16_t> total_pcm;
    for (size_t c = 0; c < chunks.size(); ++c) {
        std::vector<int32_t> ids = engine->phonemizer.phonemize(chunks[c]);
        if (ids.size() <= 2) continue;

        int vidx = static_cast<int>(ids.size()) - 1;
        if (vidx < 0) vidx = 0;
        if (vidx > 509) vidx = 509;

        std::vector<float> style(voice_pack.begin() + vidx * 256,
                                 voice_pack.begin() + (vidx + 1) * 256);

        std::vector<int16_t> chunk_pcm;
        int rc = engine->model->forward_graph(ids, style, chunk_pcm);
        if (rc == 0 && !chunk_pcm.empty()) {
            total_pcm.insert(total_pcm.end(), chunk_pcm.begin(), chunk_pcm.end());
        } else {
            LOGE("speak_raw: Chunk %zu/%zu forward_graph failed (rc=%d, tokens=%zu)", c + 1, chunks.size(), rc, ids.size());
        }
    }

    if (total_pcm.empty()) {
        LOGE("speak_raw: forward_graph produced 0 audio samples across %zu chunks", chunks.size());
        return nullptr;
    }

    // Convert int16 PCM to float [-1.0, 1.0] expected by KokoroTTS
    jsize num_samples = static_cast<jsize>(total_pcm.size());
    jfloatArray result = env->NewFloatArray(num_samples);
    if (!result) return nullptr;

    std::vector<float> float_samples(num_samples);
    for (size_t i = 0; i < total_pcm.size(); ++i) {
        float s = total_pcm[i] / 32768.0f;
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        float_samples[i] = s;
    }

    env->SetFloatArrayRegion(result, 0, num_samples, float_samples.data());
    LOGI("Generated %d audio samples (%.2fs) on Adreno GPU across %zu chunks", num_samples, num_samples / 24000.0f, chunks.size());
    return result;
}

JNIEXPORT jint JNICALL Java_com_kokoros_KokoroJNI_speak_1stream(
    JNIEnv* env,
    jclass clazz,
    jlong engine_ptr,
    jstring text,
    jstring voice,
    jstring language,
    jfloat speed,
    jobject callback
) {
    if (engine_ptr == 0) {
        LOGE("speak_stream: NULL engine pointer");
        return -1;
    }
    if (!callback) {
        LOGE("speak_stream: NULL callback");
        return -1;
    }

    auto* engine = reinterpret_cast<KokoroEngineInstance*>(engine_ptr);
    std::lock_guard<std::mutex> lock(engine->mtx);

    const char* c_text = env->GetStringUTFChars(text, nullptr);
    const char* c_voice = env->GetStringUTFChars(voice, nullptr);
    const char* c_lang = env->GetStringUTFChars(language, nullptr);

    std::string text_str = c_text ? c_text : "";
    std::string voice_str = c_voice ? c_voice : "af_heart";
    std::string lang_str = c_lang ? c_lang : "en-us";

    env->ReleaseStringUTFChars(text, c_text);
    env->ReleaseStringUTFChars(voice, c_voice);
    env->ReleaseStringUTFChars(language, c_lang);

    if (text_str.empty()) {
        return 0;
    }

    setenv("NNOPT_GPU_FP32_GENERATOR", "1", 1);

    // Get callback method ID: int audioAvailable(byte[] buffer, int offset, int length)
    jclass cb_class = env->GetObjectClass(callback);
    jmethodID mid_audioAvailable = env->GetMethodID(cb_class, "audioAvailable", "([BII)I");
    if (!mid_audioAvailable) {
        LOGE("speak_stream: audioAvailable([BII)I not found on callback");
        return -1;
    }

    // Load voice tensor
    if (engine->voice_cache.find(voice_str) == engine->voice_cache.end()) {
        std::vector<float> vdata;
        bool loaded = false;
        if (!engine->voices_bin_path.empty()) {
            loaded = read_voice_from_npz(engine->voices_bin_path, voice_str, vdata);
        }
        if (!loaded) {
            std::string fallback_voice = engine->base_dir + "/kokoro_gpu/assets/voice_pack_" + voice_str + ".bin";
            if (access(fallback_voice.c_str(), R_OK) == 0) {
                vdata = load_float_bin(fallback_voice);
                loaded = (vdata.size() == 510 * 256);
            }
        }
        if (!loaded) {
            LOGW("Could not load voice %s, trying default af_heart", voice_str.c_str());
            std::string def_voice = engine->base_dir + "/kokoro_gpu/assets/voice_pack_af_heart.bin";
            vdata = load_float_bin(def_voice);
        }
        if (vdata.size() == 510 * 256) {
            engine->voice_cache[voice_str] = std::move(vdata);
        } else {
            LOGE("Failed to find any style vector for voice: %s", voice_str.c_str());
            return -1;
        }
    }

    const auto& voice_pack = engine->voice_cache[voice_str];

    std::vector<std::string> chunks = chunk_text_by_tokens(text_str, engine->phonemizer, /*max_tokens=*/500);
    if (chunks.empty()) {
        chunks.push_back(text_str);
    }

    LOGI("Streaming synthesis: %zu chunks for '%s'", chunks.size(), text_str.c_str());

    for (size_t c = 0; c < chunks.size(); ++c) {
        std::vector<int32_t> ids = engine->phonemizer.phonemize(chunks[c]);
        if (ids.size() <= 2) {
            continue; // only pad/pad
        }

        int vidx = static_cast<int>(ids.size()) - 1;
        if (vidx < 0) vidx = 0;
        if (vidx > 509) vidx = 509;

        std::vector<float> style(voice_pack.begin() + vidx * 256,
                                 voice_pack.begin() + (vidx + 1) * 256);

        std::vector<int16_t> chunk_pcm;
        auto t0 = std::chrono::steady_clock::now();
        int rc = engine->model->forward_graph(ids, style, chunk_pcm);
        double synth_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        if (rc != 0 || chunk_pcm.empty()) {
            LOGE("Chunk %zu/%zu forward_graph failed (rc=%d)", c + 1, chunks.size(), rc);
            continue;
        }

        jsize byte_len = static_cast<jsize>(chunk_pcm.size() * sizeof(int16_t));
        jbyteArray byte_array = env->NewByteArray(byte_len);
        if (!byte_array) {
            LOGE("Failed to allocate jbyteArray (%d bytes)", byte_len);
            return -1;
        }

        env->SetByteArrayRegion(byte_array, 0, byte_len,
                                reinterpret_cast<const jbyte*>(chunk_pcm.data()));

        jint res = env->CallIntMethod(callback, mid_audioAvailable, byte_array, 0, byte_len);
        env->DeleteLocalRef(byte_array);

        double audio_s = static_cast<double>(chunk_pcm.size()) / 24000.0;
        LOGI("Stream chunk %zu/%zu: synthesized in %.1fms (%.2fs audio, RTF %.2f) -> callback returned %d",
             c + 1, chunks.size(), synth_ms, audio_s, (synth_ms / 1000.0) / audio_s, res);

        // Non-zero means stop was requested or an error occurred in the audio sink
        if (res != 0) {
            LOGI("Playback cancelled / stopped by Android (code=%d) at chunk %zu", res, c + 1);
            return 0;
        }
    }

    return 0;
}

JNIEXPORT jstring JNICALL Java_com_kokoros_KokoroJNI_get_1backend_1info(
    JNIEnv* env,
    jclass clazz,
    jlong engine_ptr
) {
    return env->NewStringUTF("Qualcomm Adreno 732 (OpenCL fp16 GPU)");
}

JNIEXPORT void JNICALL Java_com_kokoros_KokoroJNI_close(
    JNIEnv* env,
    jclass clazz,
    jlong engine_ptr
) {
    if (engine_ptr != 0) {
        LOGI("Closing Kokoro Adreno GPU Engine");
        auto* engine = reinterpret_cast<KokoroEngineInstance*>(engine_ptr);
        delete engine;
    }
}

} // extern "C"
