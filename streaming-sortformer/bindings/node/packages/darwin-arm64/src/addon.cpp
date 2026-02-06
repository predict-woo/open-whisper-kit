#include <napi.h>
#include "SortformerModel.h"
#include "StreamingSession.h"

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    SortformerModel::Init(env, exports);
    StreamingSession::Init(env, exports);
    return exports;
}

NODE_API_MODULE(sortformer, Init)
