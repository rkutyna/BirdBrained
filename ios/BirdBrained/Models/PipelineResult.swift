import Foundation
import UIKit

struct SpeciesPrediction: Identifiable {
    let id = UUID()
    let rank: Int
    let species: String
    let confidence: Float
}

struct PipelineResult: Identifiable {
    let id = UUID()
    let imageName: String
    let original: UIImage
    let crop: UIImage?
    let bbox: CGRect?            // pixel-space rect on original, nil if no detection
    let detectorConfidence: Float?
    let top5: [SpeciesPrediction]
    let sharpnessRaw: Float?     // raw tenengrad energy (batch-normalized later)
    var sharpnessScore100: Float?   // assigned after batch percentile normalization
    let error: String?

    var topSpecies: String? { top5.first?.species }
    var topConfidence: Float? { top5.first?.confidence }
}

enum ConfidenceBand {
    case high, medium, low

    static func from(_ confidence: Float?) -> ConfidenceBand? {
        guard let c = confidence else { return nil }
        if c > 0.75 { return .high }
        if c >= 0.40 { return .medium }
        return .low
    }

    var color: UIColor {
        switch self {
        case .high: .systemGreen
        case .medium: .systemYellow
        case .low: .systemRed
        }
    }

    var label: String {
        switch self {
        case .high: "high"
        case .medium: "medium"
        case .low: "low"
        }
    }
}
