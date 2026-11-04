import MLX
import MLXLMCommon
import Testing

@testable import MLXGuidedGeneration

/// A token the model already sampled must lose its lead on the next step; nothing changes without penalties.
@Test func presencePenaltyDemotesSampledToken() {
    let logits = MLXArray([Float32]([1.0, 3.0, 2.5, 0.0])).reshaped(1, 1, 4)
    #expect(GuidedPenalties.none.makeProcessor() == nil)
    var processor = GuidedPenalties(presencePenalty: 1.5).makeProcessor()
    #expect(processor != nil)
    #expect(argMax(GuidedGenerationLoop.penalized(logits, with: processor), axis: -1).item(Int32.self) == 1)
    processor?.didSample(token: MLXArray(Int32(1)))
    let penalized = GuidedGenerationLoop.penalized(logits, with: processor)
    #expect(penalized.shape == [1, 1, 4])
    #expect(argMax(penalized, axis: -1).item(Int32.self) == 2)
}
