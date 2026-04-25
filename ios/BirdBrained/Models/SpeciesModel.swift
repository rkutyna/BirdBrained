import Foundation

enum SpeciesModel: String, CaseIterable, Identifiable {
    case subset98
    case base404

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .subset98: "98 species (curated)"
        case .base404: "404 species (full NABirds)"
        }
    }

    var mlModelAssetName: String {
        switch self {
        case .subset98: "BirdClassifier98"
        case .base404: "BirdClassifier404"
        }
    }

    var classCount: Int {
        switch self {
        case .subset98: 98
        case .base404: 404
        }
    }
}
