import SwiftUI

struct ResultCardView: View {
    let result: PipelineResult

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                Image(uiImage: result.original)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 110, height: 110)
                    .clipped()
                    .clipShape(RoundedRectangle(cornerRadius: 8))

                if let crop = result.crop {
                    Image(uiImage: crop)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 110, height: 110)
                        .clipped()
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                } else {
                    ZStack {
                        RoundedRectangle(cornerRadius: 8).fill(Color(.systemGray5))
                        Text("no detection").font(.caption2).foregroundStyle(.secondary)
                    }
                    .frame(width: 110, height: 110)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text(result.imageName)
                        .font(.caption2)
                        .lineLimit(1)
                        .foregroundStyle(.secondary)
                    if let conf = result.detectorConfidence {
                        Text("Detector: \(conf, specifier: "%.2f")")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    if let s = result.sharpnessScore100 {
                        SharpnessBadge(score: s)
                    }
                }

                Spacer(minLength: 0)
            }

            if let err = result.error {
                Text(err).font(.caption).foregroundStyle(.red)
            } else if result.top5.isEmpty {
                Text("No bird detected").font(.caption).foregroundStyle(.secondary)
            } else {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(result.top5) { pred in
                        PredictionRow(prediction: pred)
                    }
                }
            }
        }
        .padding(12)
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }
}

private struct PredictionRow: View {
    let prediction: SpeciesPrediction

    var body: some View {
        let band = ConfidenceBand.from(prediction.confidence)
        HStack(spacing: 8) {
            Text("\(prediction.rank).")
                .font(.caption).foregroundStyle(.secondary).frame(width: 18, alignment: .trailing)
            Text(prediction.species)
                .font(.callout)
                .lineLimit(1)
            Spacer(minLength: 8)
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 3)
                    .fill(Color(.systemGray5))
                    .frame(width: 80, height: 6)
                RoundedRectangle(cornerRadius: 3)
                    .fill(Color(band?.color ?? .systemGray))
                    .frame(width: 80 * CGFloat(prediction.confidence), height: 6)
            }
            Text(String(format: "%.2f", prediction.confidence))
                .font(.caption)
                .foregroundStyle(Color(band?.color ?? .systemGray))
                .frame(width: 40, alignment: .trailing)
        }
    }
}

private struct SharpnessBadge: View {
    let score: Float

    var body: some View {
        let band = ConfidenceBand.from(score / 100)
        HStack(spacing: 4) {
            Image(systemName: "camera.metering.matrix").font(.caption2)
            Text("Sharpness \(Int(score.rounded()))")
                .font(.caption2)
        }
        .padding(.horizontal, 6).padding(.vertical, 3)
        .background(Color(band?.color ?? .systemGray).opacity(0.18))
        .foregroundStyle(Color(band?.color ?? .systemGray))
        .clipShape(Capsule())
    }
}
