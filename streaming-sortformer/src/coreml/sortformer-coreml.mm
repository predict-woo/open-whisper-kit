// CoreML bridge implementation for SortFormer head
//
// Objective-C++ wrapper that loads CoreML model and provides C API.

#if !__has_feature(objc_arc)
#error This file must be compiled with automatic reference counting enabled (-fobjc-arc)
#endif

#import "sortformer-coreml.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

struct sortformer_coreml_context {
    const void * model;
};

struct sortformer_coreml_context * sortformer_coreml_init(const char * path_model) {
    @autoreleasepool {
        NSString * path_str = [[NSString alloc] initWithUTF8String:path_model];
        NSURL * url_model = [NSURL fileURLWithPath:path_str];
        
        MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;
        
        NSError * error = nil;
        MLModel * model = [MLModel modelWithContentsOfURL:url_model configuration:config error:&error];
        
        if (model == nil) {
            NSLog(@"sortformer_coreml_init: failed to load model from %@: %@", path_str, error);
            return NULL;
        }
        
        struct sortformer_coreml_context * ctx = (struct sortformer_coreml_context *)malloc(sizeof(struct sortformer_coreml_context));
        if (ctx == NULL) {
            return NULL;
        }
        
        ctx->model = CFBridgingRetain(model);
        
        NSLog(@"sortformer_coreml_init: loaded CoreML model from %@", path_str);
        return ctx;
    }
}

void sortformer_coreml_free(struct sortformer_coreml_context * ctx) {
    if (ctx != NULL) {
        if (ctx->model != NULL) {
            CFRelease(ctx->model);
        }
        free(ctx);
    }
}

int sortformer_coreml_encode(
    struct sortformer_coreml_context * ctx,
    const float * pre_encoder_embs,
    int32_t       seq_len,
    float       * preds_out
) {
    if (ctx == NULL || ctx->model == NULL) {
        return -1;
    }
    
    if (seq_len > SORTFORMER_COREML_MAX_SEQ_LEN) {
        NSLog(@"sortformer_coreml_encode: seq_len %d exceeds max %d", seq_len, SORTFORMER_COREML_MAX_SEQ_LEN);
        return -1;
    }
    
    @autoreleasepool {
        NSError * error = nil;
        MLModel * model = (__bridge MLModel *)ctx->model;
        
        const int max_len = SORTFORMER_COREML_MAX_SEQ_LEN;
        const int d_model = SORTFORMER_COREML_D_MODEL;
        const int n_speakers = SORTFORMER_COREML_N_SPEAKERS;
        
        MLMultiArray * embsArray = [[MLMultiArray alloc]
            initWithShape:@[@1, @(max_len), @(d_model)]
                 dataType:MLMultiArrayDataTypeFloat32
                    error:&error];
        if (embsArray == nil) {
            NSLog(@"sortformer_coreml_encode: failed to create embs array: %@", error);
            return -1;
        }
        
        float * embsPtr = (float *)embsArray.dataPointer;
        memset(embsPtr, 0, max_len * d_model * sizeof(float));
        memcpy(embsPtr, pre_encoder_embs, seq_len * d_model * sizeof(float));
        
        // Create input: pre_encoder_lengths [1]
        MLMultiArray * lengthsArray = [[MLMultiArray alloc]
            initWithShape:@[@1]
                 dataType:MLMultiArrayDataTypeInt32
                    error:&error];
        if (lengthsArray == nil) {
            NSLog(@"sortformer_coreml_encode: failed to create lengths array: %@", error);
            return -1;
        }
        lengthsArray[0] = @(seq_len);
        
        // Create feature provider
        NSDictionary * inputDict = @{
            @"pre_encoder_embs": embsArray,
            @"pre_encoder_lengths": lengthsArray
        };
        
        MLDictionaryFeatureProvider * inputProvider = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:inputDict
                         error:&error];
        if (inputProvider == nil) {
            NSLog(@"sortformer_coreml_encode: failed to create input provider: %@", error);
            return -1;
        }
        
        // Run prediction
        id<MLFeatureProvider> output = [model predictionFromFeatures:inputProvider error:&error];
        if (output == nil) {
            NSLog(@"sortformer_coreml_encode: prediction failed: %@", error);
            return -1;
        }
        
        // Extract output: speaker_preds [1, T, n_speakers]
        MLMultiArray * predsArray = (MLMultiArray *)[output featureValueForName:@"speaker_preds"].multiArrayValue;
        if (predsArray == nil) {
            NSLog(@"sortformer_coreml_encode: missing speaker_preds output");
            return -1;
        }
        
        // Copy output (CoreML gives [1, T, 4], we want [T, 4] row-major)
        const float * predsPtr = (const float *)predsArray.dataPointer;
        memcpy(preds_out, predsPtr, seq_len * n_speakers * sizeof(float));
        
        return 0;
    }
}

#ifdef __cplusplus
}
#endif
