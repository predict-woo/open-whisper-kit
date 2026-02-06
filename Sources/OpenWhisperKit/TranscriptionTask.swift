import Foundation

enum TranscriptionTask {

    static func transcribeDirect(
        samples: [Float],
        options: DecodingOptions,
        config: OpenWhisperKitConfig,
        context: WhisperContext,
        callback: TranscriptionCallback,
        timingsStore: @Sendable (TranscriptionTimings) -> Void
    ) async throws -> TranscriptionResult {
        let inputSeconds = Double(samples.count) / AudioProcessor.sampleRate

        let bridge = CallbackBridge()
        bridge.progressCallback = callback

        try await context.fullTranscribe(
            samples: samples,
            options: options,
            config: config,
            bridge: bridge
        )

        if bridge.shouldCancel {
            throw WhisperError.cancelled
        }

        let segments = await context.getSegments(wordTimestamps: options.wordTimestamps)
        let timings = await context.getTimings(inputAudioSeconds: inputSeconds)
        let text = segments.map(\.text).joined()

        timingsStore(timings)

        return TranscriptionResult(
            text: text,
            segments: segments,
            language: segments.first.map { _ in "" } ?? "",
            timings: timings
        )
    }

    static func transcribeWithVAD(
        samples: [Float],
        options: DecodingOptions,
        config: OpenWhisperKitConfig,
        context: WhisperContext,
        vad: VADProcessor,
        callback: TranscriptionCallback,
        timingsStore: @Sendable (TranscriptionTimings) -> Void
    ) async throws -> TranscriptionResult {
        let inputSeconds = Double(samples.count) / AudioProcessor.sampleRate

        let speechSegments = try vad.detectSpeech(in: samples)

        if speechSegments.isEmpty {
            let timings = TranscriptionTimings(
                inputAudioSeconds: inputSeconds,
                modelLoadingTime: 0,
                encodingTime: 0,
                decodingTime: 0,
                fullPipeline: 0
            )
            timingsStore(timings)
            return TranscriptionResult(
                text: "",
                segments: [],
                language: "",
                timings: timings
            )
        }

        var allSegments: [TranscriptionSegment] = []
        var fullText = ""

        for speechSeg in speechSegments {
            let start = max(0, speechSeg.startSample)
            let end = min(speechSeg.endSample, samples.count)
            guard start < end else { continue }

            let chunk = Array(samples[start..<end])

            let bridge = CallbackBridge()
            bridge.progressCallback = callback

            try await context.fullTranscribe(
                samples: chunk,
                options: options,
                config: config,
                bridge: bridge
            )

            if bridge.shouldCancel {
                throw WhisperError.cancelled
            }

            var segments = await context.getSegments(wordTimestamps: options.wordTimestamps)
            let timeOffset = speechSeg.startTime

            for idx in segments.indices {
                let seg = segments[idx]
                segments[idx] = TranscriptionSegment(
                    id: allSegments.count + idx,
                    seek: seg.seek,
                    start: seg.start + timeOffset,
                    end: seg.end + timeOffset,
                    text: seg.text,
                    tokens: seg.tokens,
                    temperature: seg.temperature,
                    avgLogprob: seg.avgLogprob,
                    compressionRatio: seg.compressionRatio,
                    noSpeechProb: seg.noSpeechProb,
                    words: seg.words?.map { w in
                        WordTiming(
                            word: w.word,
                            tokens: w.tokens,
                            start: w.start + timeOffset,
                            end: w.end + timeOffset,
                            probability: w.probability
                        )
                    }
                )
            }

            allSegments.append(contentsOf: segments)
            fullText += segments.map(\.text).joined()
        }

        let timings = await context.getTimings(inputAudioSeconds: inputSeconds)
        timingsStore(timings)

        return TranscriptionResult(
            text: fullText,
            segments: allSegments,
            language: "",
            timings: timings
        )
    }
}
