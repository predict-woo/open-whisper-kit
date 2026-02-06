#include "SortformerModel.h"
#include "DiarizeWorker.h"

// Static constructor reference
Napi::FunctionReference SortformerModel::constructor;

Napi::Object SortformerModel::Init(Napi::Env env, Napi::Object exports) {
    Napi::HandleScope scope(env);
    
    Napi::Function func = DefineClass(env, "SortformerModel", {
        InstanceMethod("close", &SortformerModel::Close),
        InstanceMethod("diarize", &SortformerModel::Diarize),
    });
    
    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();
    
    exports.Set("SortformerModel", func);
    return exports;
}

SortformerModel::SortformerModel(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<SortformerModel>(info), ctx_(nullptr) {
    
    Napi::Env env = info.Env();
    
    // Validate arguments: expect exactly one string argument (model path)
    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Model path is required").ThrowAsJavaScriptException();
        return;
    }
    
    if (!info[0].IsString()) {
        Napi::TypeError::New(env, "Model path must be a string").ThrowAsJavaScriptException();
        return;
    }
    
    std::string modelPath = info[0].As<Napi::String>().Utf8Value();
    
    // Get default parameters
    sortformer_params params = sortformer_default_params();
    
    // Initialize the sortformer context
    ctx_ = sortformer_init(modelPath.c_str(), params);
    
    if (ctx_ == nullptr) {
        Napi::Error::New(env, "Failed to load model from path: " + modelPath).ThrowAsJavaScriptException();
        return;
    }
}

SortformerModel::~SortformerModel() {
    Cleanup();
}

void SortformerModel::Cleanup() {
    if (ctx_ != nullptr) {
        sortformer_free(ctx_);
        ctx_ = nullptr;
    }
}

Napi::Value SortformerModel::Close(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    Cleanup();
    
    return env.Undefined();
}

Napi::Value SortformerModel::Diarize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (ctx_ == nullptr) {
        Napi::Error::New(env, "Model is closed or not initialized").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Audio data is required").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    if (!info[0].IsTypedArray()) {
        Napi::TypeError::New(env, "Audio must be a Float32Array").ThrowAsJavaScriptException();
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
        Napi::Error::New(env, "Audio data cannot be empty").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    std::vector<float> audio(audioLength);
    for (size_t i = 0; i < audioLength; i++) {
        audio[i] = audioArray[i];
    }
    
    DiarizeOptions options;
    
    if (info.Length() >= 2 && info[1].IsObject()) {
        Napi::Object opts = info[1].As<Napi::Object>();
        
        if (opts.Has("threshold") && opts.Get("threshold").IsNumber()) {
            options.threshold = opts.Get("threshold").As<Napi::Number>().FloatValue();
        }
        
        if (opts.Has("medianFilter") && opts.Get("medianFilter").IsNumber()) {
            options.median_filter = opts.Get("medianFilter").As<Napi::Number>().Int32Value();
        }
        
        if (opts.Has("filename") && opts.Get("filename").IsString()) {
            options.filename = opts.Get("filename").As<Napi::String>().Utf8Value();
        }
    }
    
    Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
    
    DiarizeWorker* worker = new DiarizeWorker(
        env,
        ctx_,
        std::move(audio),
        options,
        deferred);
    
    worker->Queue();
    
    return deferred.Promise();
}
