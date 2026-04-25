import UIKit
import Vision
import CoreML

/// Wraps the YOLO11n CoreML export (NMS baked in). Returns the highest-confidence "bird" detection.
/// Swift auto-generates a class `BirdDetector` from `BirdDetector.mlpackage` after you drag it into
/// the Xcode project — that is what we instantiate below.
final class Detector {
    struct Detection {
        let boundingBoxPixels: CGRect    // origin top-left, on the original image
        let confidence: Float
    }

    private let vnModel: VNCoreMLModel
    private let confidenceThreshold: Float

    init(confidenceThreshold: Float = 0.25) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let core = try BirdDetector(configuration: config)
        self.vnModel = try VNCoreMLModel(for: core.model)
        self.confidenceThreshold = confidenceThreshold
    }

    /// Runs detection on an orientation-normalized UIImage and returns the best bird, if any.
    func detectBestBird(in image: UIImage) async throws -> Detection? {
        guard let cg = image.cgImage else { return nil }

        let request = VNCoreMLRequest(model: vnModel)
        request.imageCropAndScaleOption = .scaleFit
        let handler = VNImageRequestHandler(cgImage: cg, orientation: .up, options: [:])

        try handler.perform([request])

        guard let observations = request.results as? [VNRecognizedObjectObservation] else {
            return nil
        }

        var best: VNRecognizedObjectObservation?
        var bestConf: Float = 0
        for obs in observations {
            guard let top = obs.labels.first else { continue }
            if top.identifier.lowercased() != "bird" { continue }
            let c = top.confidence
            if c < confidenceThreshold { continue }
            if best == nil || c > bestConf {
                best = obs
                bestConf = c
            }
        }

        guard let match = best else { return nil }

        let imageSize = CGSize(width: cg.width, height: cg.height)
        let rect = CropUtils.pixelRectFromVision(normalizedBox: match.boundingBox,
                                                  imageSize: imageSize)
        return Detection(boundingBoxPixels: rect, confidence: bestConf)
    }
}
