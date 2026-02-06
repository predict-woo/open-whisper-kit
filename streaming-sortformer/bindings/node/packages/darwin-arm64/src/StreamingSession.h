#ifndef STREAMING_SESSION_H
#define STREAMING_SESSION_H

#include <napi.h>
#include "sortformer.h"

class StreamingSession : public Napi::ObjectWrap<StreamingSession> {
public:
    // Initialize the class and register with N-API
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    
    // Constructor - takes SortformerModel and preset
    StreamingSession(const Napi::CallbackInfo& info);
    
    // Destructor
    ~StreamingSession();

private:
    // Static constructor reference for N-API
    static Napi::FunctionReference constructor;
    
    // The underlying streaming state (owned)
    sortformer_stream_state* stream_;
    
    // Reference to the model context (not owned - must outlive session)
    sortformer_context* ctx_;
    
    // Whether the session has been closed
    bool closed_;
    
    // Total frames output so far
    int64_t total_frames_;
    
    // Feed audio samples, get predictions
    Napi::Value Feed(const Napi::CallbackInfo& info);
    
    // Flush remaining buffered audio
    Napi::Value Flush(const Napi::CallbackInfo& info);
    
    // Reset streaming state
    Napi::Value Reset(const Napi::CallbackInfo& info);
    
    // Close and free resources
    Napi::Value Close(const Napi::CallbackInfo& info);
    
    // Get total frames output
    Napi::Value GetTotalFrames(const Napi::CallbackInfo& info);
    
    // Check if closed
    Napi::Value IsClosed(const Napi::CallbackInfo& info);
    
    // Internal cleanup
    void Cleanup();
};

#endif // STREAMING_SESSION_H
