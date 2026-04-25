# BirdBrained iOS — Xcode setup

The Swift source tree is written as plain files under `ios/BirdBrained/`. Follow these steps in Xcode to wire them into a buildable iOS app.

## 1. Create the Xcode project

1. Xcode → **File → New → Project…**
2. iOS → **App** → Next
3. Fill in:
   - Product Name: `BirdBrained`
   - Team: your free Apple ID
   - Organization Identifier: any reverse-domain you own (e.g. `com.yourname`)
   - Interface: **SwiftUI**
   - Language: **Swift**
   - Storage: **None**
   - Include Tests: off (optional)
4. Save the project inside `ios/` (alongside the existing `BirdBrained/` folder). Let Xcode create `ios/BirdBrained.xcodeproj`.

The default template will also create a new `ios/BirdBrained/` folder containing `BirdBrainedApp.swift`, `ContentView.swift`, `Assets.xcassets`, etc. **Delete that template `BirdBrainedApp.swift` and `ContentView.swift`** (move to trash) — we're replacing them with the hand-written tree. Keep `Assets.xcassets` and the `Preview Content` group.

## 2. Add our Swift sources

1. In Project Navigator, right-click the `BirdBrained` group → **Add Files to "BirdBrained"…**
2. Navigate to `ios/BirdBrained/` in the repo.
3. Select these folders (hold ⌘ to multi-select):
   - `Views`
   - `Pipeline`
   - `Models`
   - `BirdBrainedApp.swift` (the one in this folder, not the template's)
4. In the dialog:
   - **Create groups** (not folder references)
   - **Copy items if needed**: leave OFF (keep source-of-truth in the repo tree)
   - Add to target: **BirdBrained**
5. Click Add.

## 3. Add the CoreML models

1. Still in Project Navigator, right-click `BirdBrained` → **Add Files…**
2. Navigate to `artifacts/coreml/` in the repo.
3. Select all three `.mlpackage` bundles:
   - `BirdClassifier98.mlpackage`
   - `BirdClassifier404.mlpackage`
   - `BirdDetector.mlpackage`
4. Dialog settings:
   - **Copy items if needed**: ON (so the models ship inside the app bundle)
   - Add to target: **BirdBrained**
5. Click Add.

Xcode will auto-generate a Swift class for each model. Verify: open the Project Navigator → click `BirdClassifier98.mlpackage` → Xcode model viewer. In the top bar, open **Model Class** — the generated class name should be `BirdClassifier98`. Likewise for `BirdClassifier404` and `BirdDetector`. These names are what `Classifier.swift` and `Detector.swift` reference.

## 4. Project settings

Select the project → **BirdBrained** target → **General** tab:
- **Minimum Deployments**: iOS **18.0**
- **Identity**: set Bundle Identifier to something unique (e.g. `com.yourname.birdbrained`)
- **Signing & Capabilities**: choose your personal Team (free Apple ID works)

**Build Settings** tab → search for `Swift Language Version` → set to **Swift 6** or **Swift 5** (either works). If Swift 6 surfaces strict-concurrency warnings, switch to Swift 5 for the MVP.

## 5. Build & run

- Plug in your iPhone (or pick a simulator with iOS 18).
- `⌘R`. First build will take ~1 min (model compilation).
- The app opens to the gallery screen. Tap **Pick photos**, select a few from the simulator's Photos app (or a real device library), and watch results populate.

To swap species models, tap the ⚙ gear icon → choose 98 vs 404. The next batch you process will use the selected model.

## 6. Troubleshooting

- **"Cannot find type 'BirdDetector' in scope"** — the .mlpackage wasn't added to the target. Project Navigator → click the file → File Inspector (right pane) → ensure **Target Membership → BirdBrained** is checked.
- **"No such module 'PhotosUI'"** — this module is part of iOS 16+. Confirm deployment target is iOS 18.
- **Black crop / empty prediction** — likely an image orientation issue. `CropUtils.normalizedOrientation` should fix most cases; if not, inspect the `detection.boundingBoxPixels` in the debugger.
- **Signing error on device** — you need to trust your developer certificate: on the iPhone, Settings → General → VPN & Device Management → trust the certificate under your Apple ID.

## Files in this tree

```
BirdBrainedApp.swift       # @main SwiftUI app entry
Views/
  GalleryView.swift         # photo picker + scrolling results
  ResultCardView.swift      # per-photo card (original, crop, top-5, sharpness)
  SettingsView.swift        # species model picker
Pipeline/
  BirdPipeline.swift        # orchestrator (actor) + batch sharpness normalizer
  Detector.swift            # YOLO11n CoreML wrapper via Vision
  Classifier.swift          # BirdClassifier{98,404} CoreML wrapper via Vision
  Sharpness.swift           # Tenengrad Sobel + patch top-5 + percentile bounds
  CropUtils.swift           # 240×240 resize-pad, orientation normalization, bbox conversion
Models/
  PipelineResult.swift      # result row + SpeciesPrediction + ConfidenceBand
  SpeciesModel.swift        # subset98 / base404 enum
```
