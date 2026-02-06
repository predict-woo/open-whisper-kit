#include "StreamingSession.h"
#include "SortformerModel.h"

Napi::FunctionReference StreamingSession::constructor;

Napi::Object StreamingSession::Init(Napi::Env env, Napi::Object exports) {
    Napi::HandleScope scope(env);
    
    Napi::Function func = DefineClass(env, "StreamingSession", {
        InstanceMethod("feed", &StreamingSession::Feed),
        InstanceMethod("flush", &StreamingSession::Flush),
        InstanceMethod("reset", &StreamingSession::Reset),
        InstanceMethod("close", &StreamingSession::Close),
        InstanceMethod("getTotalFrames", &StreamingSession::GetTotalFrames),
        InstanceMethod("isClosed", &StreamingSession::IsClosed),
    });
    
    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();
    
    exports.Set("StreamingSession", func);
    return exports;
}

StreamingSession::StreamingSession(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<StreamingSession>(info), stream_(nullptr), ctx_(nullptr), closed_(false), total_frames_(0) {
    
    Napi::Env env = info.Env();
    
    // Expect: new StreamingSession(model, preset)
    // model: SortformerModel instance
    // preset: number (0=low, 1=2s, 2=3s, 3=5s)
    
    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected (model, preset) arguments").ThrowAsJavaScriptException();
        return;
    }
    
    // Get model context from SortformerModel
    if (!info[0].IsObject()) {
        Napi::TypeError::New(env, "First argument must be a SortformerModel").ThrowAsJavaScriptException();
        return;
    }
    
    Napi::Object modelObj = info[0].As<Napi::Object>();
    SortformerModel* model = Napi::ObjectWrap<SortformerModel>::Unwrap(modelObj);
    
    if (!model || !model->IsValid()) {
        Napi::Error::New(env, "Model is closed or invalid").ThrowAsJavaScriptException();
        return;
    }
    
    ctx_ = model->GetContext();
    
    // Get preset
    if (!info[1].IsNumber()) {
        Napi::TypeError::New(env, "Second argument must be a preset number").ThrowAsJavaScriptException();
        return;
    }
    
    int preset_num = info[1].As<Napi::Number>().Int32Value();
    sortformer_stream_preset preset = static_cast<sortformer_stream_preset>(preset_num);
    
    // Initialize streaming session
    stream_ = sortformer_stream_init(ctx_, preset);
    
    if (stream_ == nullptr) {
        Napi::Error::New(env, "Failed to create streaming session").ThrowAsJavaScriptException();
        return;
    }
}

StreamingSession::~StreamingSession() {
    Cleanup();
}

void StreamingSession::Cleanup() {
    if (stream_ != nullptr) {
        sortformer_stream_free(stream_);
        stream_ = nullptr;
    }
    closed_ = true;
}

Napi::Value StreamingSession::Feed(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (closed_ || stream_ == nullptr) {
        Napi::Error::New(env, "Session is closed").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    if (info.Length() < 1 || !info[0].IsTypedArray()) {
        Napi::TypeError::New(env, "Expected Float32Array argument").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    Napi::TypedArray typedArray = info[0].As<Napi::TypedArray>();
    if (typedArray.TypedArrayType() != napi_float32_array) {
        Napi::TypeError::New(env, "Audio must be a Float32Array").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    Napi::Float32Array audioArray = info[0].As<Napi::Float32Array>();
    size_t audioLength = audioArray.ElementLength();
    
    if (audioLength == 0) {
        // Return empty result
        Napi::Object result = Napi::Object::New(env);
        result.Set("predictions", Napi::Float32Array::New(env, 0));
        result.Set("frameCount", Napi::Number::New(env, 0));
        return result;
    }
    
    // Get audio data pointer
    float* audioData = audioArray.Data();
    
    // Allocate output buffer (generous size)
    int max_frames = (audioLength / 160) + 100;  // hop=160, plus margin
    std::vector<float> probs_out(max_frames * 4);
    
    // Feed to streaming pipeline
    int n_frames = sortformer_stream_feed(stream_, audioData, audioLength, 
                                           probs_out.data(), max_frames);
    
    if (n_frames < 0) {
        Napi::Error::New(env, "Stream feed failed").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    total_frames_ += n_frames;
    
    // Create result object
    Napi::Object result = Napi::Object::New(env);
    
    // Copy predictions to Float32Array
    Napi::Float32Array predictions = Napi::Float32Array::New(env, n_frames * 4);
    for (int i = 0; i < n_frames * 4; i++) {
        predictions[i] = probs_out[i];
    }
    
    result.Set("predictions", predictions);
    result.Set("frameCount", Napi::Number::New(env, n_frames));
    
    return result;
}

Napi::Value StreamingSession::Flush(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (closed_ || stream_ == nullptr) {
        Napi::Error::New(env, "Session is closed").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    int max_frames = 1000;
    std::vector<float> probs_out(max_frames * 4);
    
    int n_frames = sortformer_stream_flush(stream_, probs_out.data(), max_frames);
    
    if (n_frames < 0) {
        Napi::Error::New(env, "Stream flush failed").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    total_frames_ += n_frames;
    
    Napi::Object result = Napi::Object::New(env);
    Napi::Float32Array predictions = Napi::Float32Array::New(env, n_frames * 4);
    for (int i = 0; i < n_frames * 4; i++) {
        predictions[i] = probs_out[i];
    }
    
    result.Set("predictions", predictions);
    result.Set("frameCount", Napi::Number::New(env, n_frames));
    
    return result;
}

Napi::Value StreamingSession::Reset(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (closed_ || stream_ == nullptr) {
        Napi::Error::New(env, "Session is closed").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    sortformer_stream_reset(stream_);
    total_frames_ = 0;
    
    return env.Undefined();
}

Napi::Value StreamingSession::Close(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    Cleanup();
    
    return env.Undefined();
}

Napi::Value StreamingSession::GetTotalFrames(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return Napi::Number::New(env, static_cast<double>(total_frames_));
}

Napi::Value StreamingSession::IsClosed(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return Napi::Boolean::New(env, closed_);
}
