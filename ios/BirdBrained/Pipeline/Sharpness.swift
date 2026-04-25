import UIKit
import Accelerate

enum Sharpness {
    /// Tenengrad sharpness on the detection crop. Ports `tenengrad_sharpness` in bird_pipeline.py.
    /// Returns the mean of the top-5 patch energies; patches are 10% of width with 50% overlap.
    static func tenengrad(_ image: UIImage) -> Float {
        guard let gray = grayscaleFloatBuffer(image),
              gray.width >= 3, gray.height >= 3 else {
            return 0.0
        }

        let energy = sobelEnergyReflect(gray: gray)
        let h = energy.height
        let w = energy.width

        var window = max(3, Int((0.10 * Double(w)).rounded()))
        window = min(window, w, h)
        let stride = max(1, Int((0.50 * Double(window)).rounded()))

        let xs = slidingStarts(length: w, window: window, stride: stride)
        let ys = slidingStarts(length: h, window: window, stride: stride)

        var patchScores: [Float] = []
        patchScores.reserveCapacity(xs.count * ys.count)

        for y in ys {
            for x in xs {
                patchScores.append(patchMean(energy: energy, x: x, y: y, w: window, h: window))
            }
        }

        if patchScores.isEmpty {
            var total: Float = 0
            vDSP_meanv(energy.data, 1, &total, vDSP_Length(energy.data.count))
            return total
        }

        let k = min(5, patchScores.count)
        let topK = patchScores.sorted(by: >).prefix(k)
        let sum = topK.reduce(Float(0), +)
        return sum / Float(k)
    }

    // MARK: - helpers

    struct FloatPlane {
        let data: [Float]
        let width: Int
        let height: Int
    }

    private static func grayscaleFloatBuffer(_ image: UIImage) -> FloatPlane? {
        guard let cg = image.cgImage else { return nil }
        let w = cg.width
        let h = cg.height
        if w == 0 || h == 0 { return nil }

        // Render into an 8-bit grayscale context.
        let colorSpace = CGColorSpaceCreateDeviceGray()
        var bytes = [UInt8](repeating: 0, count: w * h)
        guard let ctx = CGContext(data: &bytes,
                                  width: w,
                                  height: h,
                                  bitsPerComponent: 8,
                                  bytesPerRow: w,
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue) else {
            return nil
        }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))

        var floats = [Float](repeating: 0, count: w * h)
        vDSP_vfltu8(bytes, 1, &floats, 1, vDSP_Length(w * h))
        return FloatPlane(data: floats, width: w, height: h)
    }

    /// Compute gx² + gy² using Sobel kernels with reflect-1 padding, matching the Python routine.
    private static func sobelEnergyReflect(gray: FloatPlane) -> FloatPlane {
        let w = gray.width
        let h = gray.height
        var energy = [Float](repeating: 0, count: w * h)

        @inline(__always)
        func pixel(_ x: Int, _ y: Int) -> Float {
            // True reflect (matches numpy mode="reflect"): index -1 → 1, index N → N-2.
            let xi: Int
            if x < 0 { xi = -x }
            else if x >= w { xi = 2 * (w - 1) - x }
            else { xi = x }
            let yi: Int
            if y < 0 { yi = -y }
            else if y >= h { yi = 2 * (h - 1) - y }
            else { yi = y }
            return gray.data[yi * w + xi]
        }

        for y in 0..<h {
            for x in 0..<w {
                let p00 = pixel(x - 1, y - 1)
                let p01 = pixel(x,     y - 1)
                let p02 = pixel(x + 1, y - 1)
                let p10 = pixel(x - 1, y    )
                let p12 = pixel(x + 1, y    )
                let p20 = pixel(x - 1, y + 1)
                let p21 = pixel(x,     y + 1)
                let p22 = pixel(x + 1, y + 1)

                let gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22
                let gy =  p00 + 2 * p01 + p02 - p20 - 2 * p21 - p22
                energy[y * w + x] = gx * gx + gy * gy
            }
        }
        return FloatPlane(data: energy, width: w, height: h)
    }

    private static func slidingStarts(length: Int, window: Int, stride: Int) -> [Int] {
        if length <= window { return [0] }
        var starts = Array(Swift.stride(from: 0, through: length - window, by: stride))
        if starts.isEmpty || starts.last! != length - window {
            starts.append(length - window)
        }
        return starts
    }

    private static func patchMean(energy: FloatPlane, x: Int, y: Int, w: Int, h: Int) -> Float {
        var sum: Float = 0
        for row in 0..<h {
            let rowStart = (y + row) * energy.width + x
            var rowSum: Float = 0
            energy.data.withUnsafeBufferPointer { ptr in
                vDSP_sve(ptr.baseAddress!.advanced(by: rowStart), 1, &rowSum, vDSP_Length(w))
            }
            sum += rowSum
        }
        return sum / Float(w * h)
    }

    // MARK: - batch percentile normalization

    /// Mirrors `sharpness_percentile_bounds` + `normalize_tenengrad_score` from bird_pipeline.py.
    static func percentileBounds(_ rawScores: [Float], lowPct: Float = 5.0, highPct: Float = 95.0) -> (low: Float, high: Float) {
        let finite = rawScores.filter { $0.isFinite }
        guard !finite.isEmpty else { return (0.0, 1.0) }
        let sorted = finite.sorted()
        let low = percentile(sortedAscending: sorted, pct: lowPct)
        var high = percentile(sortedAscending: sorted, pct: highPct)
        if !low.isFinite || !high.isFinite || high <= low {
            high = low + 1e-6
        }
        return (low, high)
    }

    private static func percentile(sortedAscending a: [Float], pct: Float) -> Float {
        // Linear interpolation between closest ranks, matching numpy.percentile default.
        let n = a.count
        if n == 1 { return a[0] }
        let rank = Double(pct) / 100.0 * Double(n - 1)
        let lo = Int(rank.rounded(.down))
        let hi = min(lo + 1, n - 1)
        let frac = Float(rank - Double(lo))
        return a[lo] * (1 - frac) + a[hi] * frac
    }

    static func normalize(raw: Float, low: Float, high: Float) -> Float {
        guard raw.isFinite else { return 0.0 }
        if raw <= low { return 0.0 }
        if raw >= high { return 100.0 }
        let span = max(1e-12, high - low)
        return max(0, min(100, 100.0 * (raw - low) / span))
    }
}
