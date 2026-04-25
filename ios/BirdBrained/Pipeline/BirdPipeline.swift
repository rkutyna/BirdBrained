import UIKit

/// Orchestrates detect → crop → sharpness → resize-pad → classify for a single image.
/// Matches `run_inference_batch` in bird_pipeline.py at a per-image level.
actor BirdPipeline {
    private let detector: Detector
    private var classifier: Classifier
    private(set) var currentModel: SpeciesModel

    init(speciesModel: SpeciesModel) throws {
        self.detector = try Detector()
        self.classifier = try Classifier(model: speciesModel)
        self.currentModel = speciesModel
    }

    func switchModel(to model: SpeciesModel) throws {
        if model == currentModel { return }
        classifier = try Classifier(model: model)
        currentModel = model
    }

    /// Run the full pipeline on one image. Returns a result with `sharpnessScore100 == nil`;
    /// caller normalizes sharpness across the batch afterward.
    func process(image input: UIImage, imageName: String) async -> PipelineResult {
        let normalized = CropUtils.normalizedOrientation(input)

        do {
            let detection = try await detector.detectBestBird(in: normalized)
            guard let det = detection else {
                return PipelineResult(imageName: imageName,
                                      original: normalized,
                                      crop: nil,
                                      bbox: nil,
                                      detectorConfidence: nil,
                                      top5: [],
                                      sharpnessRaw: nil,
                                      sharpnessScore100: nil,
                                      error: nil)
            }

            guard let crop = CropUtils.cropFullResolution(normalized, to: det.boundingBoxPixels) else {
                return PipelineResult(imageName: imageName,
                                      original: normalized,
                                      crop: nil,
                                      bbox: det.boundingBoxPixels,
                                      detectorConfidence: det.confidence,
                                      top5: [],
                                      sharpnessRaw: nil,
                                      sharpnessScore100: nil,
                                      error: "degenerate crop")
            }

            let sharpnessRaw = Sharpness.tenengrad(crop)
            let padded = CropUtils.resizePad(crop)
            let top5 = try await classifier.classify(padded)

            return PipelineResult(imageName: imageName,
                                  original: normalized,
                                  crop: crop,
                                  bbox: det.boundingBoxPixels,
                                  detectorConfidence: det.confidence,
                                  top5: top5,
                                  sharpnessRaw: sharpnessRaw,
                                  sharpnessScore100: nil,
                                  error: nil)
        } catch {
            return PipelineResult(imageName: imageName,
                                  original: normalized,
                                  crop: nil,
                                  bbox: nil,
                                  detectorConfidence: nil,
                                  top5: [],
                                  sharpnessRaw: nil,
                                  sharpnessScore100: nil,
                                  error: error.localizedDescription)
        }
    }
}

/// Apply the batch 5th/95th-percentile normalization from `run_inference_batch`.
enum SharpnessBatchNormalizer {
    static func normalize(_ results: inout [PipelineResult]) {
        let raws = results.compactMap { $0.sharpnessRaw }
        guard !raws.isEmpty else { return }
        let bounds = Sharpness.percentileBounds(raws)
        for i in results.indices {
            if let raw = results[i].sharpnessRaw {
                results[i].sharpnessScore100 = Sharpness.normalize(raw: raw,
                                                                   low: bounds.low,
                                                                   high: bounds.high)
            }
        }
    }
}
