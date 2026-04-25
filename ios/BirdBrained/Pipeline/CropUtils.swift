import UIKit
import CoreGraphics

// Matches IMAGENET_PAD_RGB = tuple(int(round(c * 255)) for c in IMAGENET_MEAN)
// where IMAGENET_MEAN = (0.485, 0.456, 0.406)
let imageNetPadColor = UIColor(red: 124.0/255.0,
                               green: 116.0/255.0,
                               blue: 104.0/255.0,
                               alpha: 1.0)

let classifierInputSize: CGFloat = 240

enum CropUtils {
    /// Crop the UIImage to the given pixel-space rect (origin top-left).
    static func cropFullResolution(_ image: UIImage, to pixelRect: CGRect) -> UIImage? {
        guard let cg = image.cgImage else { return nil }

        let w = CGFloat(cg.width)
        let h = CGFloat(cg.height)

        var r = pixelRect
        r.origin.x = max(0, min(r.origin.x, w))
        r.origin.y = max(0, min(r.origin.y, h))
        r.size.width = max(0, min(r.size.width, w - r.origin.x))
        r.size.height = max(0, min(r.size.height, h - r.origin.y))

        if r.width <= 0 || r.height <= 0 { return nil }

        // cgImage is oriented with origin top-left regardless of UIImage.orientation — we already
        // baked-in the orientation upstream by redrawing via normalizedOrientation().
        guard let cropped = cg.cropping(to: r) else { return nil }
        return UIImage(cgImage: cropped, scale: image.scale, orientation: .up)
    }

    /// Resize the image to fit within `size`×`size`, preserving aspect ratio via BILINEAR,
    /// then center on a square canvas filled with `padColor`. Mirrors Python's `crop_resize_pad`.
    static func resizePad(_ image: UIImage,
                          size: CGFloat = classifierInputSize,
                          padColor: UIColor = imageNetPadColor) -> UIImage {
        let srcW = image.size.width
        let srcH = image.size.height
        let scale = min(size / srcW, size / srcH)
        let newW = max(1, (srcW * scale).rounded())
        let newH = max(1, (srcH * scale).rounded())

        let fmt = UIGraphicsImageRendererFormat()
        fmt.scale = 1   // render at 1:1 pixel mapping — CoreML expects 240×240 pixels
        fmt.opaque = true

        let renderer = UIGraphicsImageRenderer(size: CGSize(width: size, height: size), format: fmt)
        return renderer.image { ctx in
            padColor.setFill()
            ctx.fill(CGRect(x: 0, y: 0, width: size, height: size))

            let left = (size - newW) / 2
            let top = (size - newH) / 2
            // High-quality scaling ≈ bilinear at this step.
            ctx.cgContext.interpolationQuality = .high
            image.draw(in: CGRect(x: left, y: top, width: newW, height: newH))
        }
    }

    /// Return a UIImage redrawn so pixel data matches the `.up` orientation.
    /// Vision handles EXIF internally, but we need the displayed orientation to match the pixel grid
    /// we crop out of — otherwise bboxes land in the wrong place.
    static func normalizedOrientation(_ image: UIImage) -> UIImage {
        if image.imageOrientation == .up { return image }
        let fmt = UIGraphicsImageRendererFormat()
        fmt.scale = image.scale
        let renderer = UIGraphicsImageRenderer(size: image.size, format: fmt)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: image.size))
        }
    }

    /// Convert a Vision-normalized bbox (origin bottom-left) to pixel-space rect (origin top-left).
    static func pixelRectFromVision(normalizedBox: CGRect, imageSize: CGSize) -> CGRect {
        let w = imageSize.width
        let h = imageSize.height
        let x = normalizedBox.minX * w
        let y = (1.0 - normalizedBox.maxY) * h
        return CGRect(x: x,
                      y: y,
                      width: normalizedBox.width * w,
                      height: normalizedBox.height * h)
    }
}
