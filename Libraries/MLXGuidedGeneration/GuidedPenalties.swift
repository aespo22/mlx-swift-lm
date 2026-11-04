import MLX
import MLXLMCommon

/// Logit penalties for grammar-constrained decoding.
///
/// Guided generation always takes the argmax, so a model that starts repeating one list item keeps
/// repeating it until the token budget runs out and the JSON comes back truncated (Qwen3.5-4B on
/// 14K-character Italian transcripts, 2026-11-04). These penalties reuse the unconstrained sampler's
/// `PenaltyProcessor` arithmetic on the tokens the model sampled in the current generation only:
/// not on the prompt, whose content the output is meant to restate, and not on grammar-forced
/// (fast-forward) tokens, which the model never chose. Default: no penalty.
public struct GuidedPenalties: Sendable, Equatable {
    public static let none = GuidedPenalties()

    /// Multiplicative penalty on recent tokens (> 1 discourages them), as `GenerateParameters.repetitionPenalty`.
    public var repetitionPenalty: Float?
    public var repetitionContextSize: Int
    /// Additive penalty subtracted once from every token seen in the recent window, as `GenerateParameters.presencePenalty`.
    public var presencePenalty: Float?
    public var presenceContextSize: Int

    public init(
        repetitionPenalty: Float? = nil, repetitionContextSize: Int = 64,
        presencePenalty: Float? = nil, presenceContextSize: Int = 128
    ) {
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.presencePenalty = presencePenalty
        self.presenceContextSize = presenceContextSize
    }

    func makeProcessor() -> PenaltyProcessor? {
        let repetition = repetitionPenalty.flatMap { $0 != 0 && repetitionContextSize > 0
            ? RepetitionContext(repetitionPenalty: $0, repetitionContextSize: repetitionContextSize) : nil }
        let presence = presencePenalty.flatMap { $0 != 0 && presenceContextSize > 0
            ? PresencePenaltyContext(presencePenalty: $0, presenceContextSize: presenceContextSize) : nil }
        guard repetition != nil || presence != nil else { return nil }
        return PenaltyProcessor(repetitionContext: repetition, presenceContext: presence, frequencyContext: nil)
    }
}

extension GuidedGenerationLoop {
    /// The last position's logits with `processor`'s penalties applied, shaped `[batch, 1, vocab]`
    /// so `applyMaskAndSample` reads them like raw model output.
    static func penalized(_ logits: MLXArray, with processor: PenaltyProcessor?) -> MLXArray {
        guard let processor else { return logits }
        return processor.process(logits: logits[0..., -1, 0...]).expandedDimensions(axis: 1)
    }
}
