#ifndef SORTFORMER_MODEL_H
#define SORTFORMER_MODEL_H

#include <napi.h>
#include "sortformer.h"

class SortformerModel : public Napi::ObjectWrap<SortformerModel> {
public:
    // Initialize the class and register with N-API
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    
    // Constructor - takes model path string
    SortformerModel(const Napi::CallbackInfo& info);
    
    // Destructor - frees sortformer context
    ~SortformerModel();
    
    // Get the underlying context pointer (for DiarizeWorker)
    sortformer_context* GetContext() { return ctx_; }
    
    // Check if context is valid
    bool IsValid() const { return ctx_ != nullptr; }

private:
    // Static constructor reference for N-API
    static Napi::FunctionReference constructor;
    
    // The underlying sortformer context
    sortformer_context* ctx_;
    
    // Explicit cleanup method (callable from JavaScript)
    Napi::Value Close(const Napi::CallbackInfo& info);
    
    // Diarization method - runs async inference via DiarizeWorker
    Napi::Value Diarize(const Napi::CallbackInfo& info);
    
    // Internal cleanup helper
    void Cleanup();
};

#endif // SORTFORMER_MODEL_H
