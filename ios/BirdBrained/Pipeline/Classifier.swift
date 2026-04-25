import UIKit
import Vision
import CoreML

/// Runs the BirdClassifier CoreML model (ResNet-50 + GeM; ImageNet normalization + softmax baked in
/// at export time, plus ClassifierConfig for label lookup — so Vision returns labels directly).
final class Classifier {
    private let vnModel: VNCoreMLModel
    let speciesModel: SpeciesModel

    init(model: SpeciesModel) throws {
        self.speciesModel = model
        let config = MLModelConfiguration()
        config.computeUnits = .all

        let coreModel: MLModel
        switch model {
        case .subset98:
            coreModel = try BirdClassifier98(configuration: config).model
        case .base404:
            coreModel = try BirdClassifier404(configuration: config).model
        }
        self.vnModel = try VNCoreMLModel(for: coreModel)
    }

    /// Classify a 240×240 padded crop. Returns top-5 (species, probability).
    func classify(_ padded240: UIImage) async throws -> [SpeciesPrediction] {
        guard let cg = padded240.cgImage else { return [] }

        let request = VNCoreMLRequest(model: vnModel)
        // Image is already at the exact input size — keep it centered with no further scaling.
        request.imageCropAndScaleOption = .centerCrop
        let handler = VNImageRequestHandler(cgImage: cg, orientation: .up, options: [:])
        try handler.perform([request])

        guard let observations = request.results as? [VNClassificationObservation] else {
            return []
        }

        let top5 = observations.prefix(5).enumerated().map { (idx, obs) in
            SpeciesPrediction(rank: idx + 1,
                              species: obs.identifier,
                              confidence: obs.confidence)
        }
        return Array(top5)
    }
}
