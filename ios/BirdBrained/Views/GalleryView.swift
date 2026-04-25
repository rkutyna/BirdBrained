import SwiftUI
import PhotosUI

@MainActor
final class GalleryViewModel: ObservableObject {
    @Published var results: [PipelineResult] = []
    @Published var isProcessing = false
    @Published var progress: String = ""
    @Published var errorMessage: String?

    private var pipeline: BirdPipeline?
    private var activeModel: SpeciesModel?

    func ensurePipeline(for model: SpeciesModel) async {
        do {
            if pipeline == nil {
                pipeline = try BirdPipeline(speciesModel: model)
                activeModel = model
            } else if activeModel != model, let p = pipeline {
                try await p.switchModel(to: model)
                activeModel = model
            }
        } catch {
            errorMessage = "Failed to load models: \(error.localizedDescription)"
        }
    }

    func process(selection: [PhotosPickerItem], model: SpeciesModel) async {
        guard !selection.isEmpty else { return }
        isProcessing = true
        results = []
        errorMessage = nil
        progress = "Loading models..."

        await ensurePipeline(for: model)
        guard let pipeline else {
            isProcessing = false
            return
        }

        let total = selection.count
        var loaded: [(UIImage, String)] = []
        for (i, item) in selection.enumerated() {
            progress = "Loading photo \(i + 1)/\(total)"
            do {
                if let data = try await item.loadTransferable(type: Data.self),
                   let image = UIImage(data: data) {
                    let name = item.itemIdentifier ?? "photo_\(i + 1)"
                    loaded.append((image, name))
                }
            } catch {
                // swallow: skip unreadable items
            }
        }

        var finished: [PipelineResult] = []
        for (i, pair) in loaded.enumerated() {
            progress = "Processing \(i + 1)/\(loaded.count): \(pair.1)"
            let r = await pipeline.process(image: pair.0, imageName: pair.1)
            finished.append(r)
            results = finished   // incremental reveal in UI
        }

        SharpnessBatchNormalizer.normalize(&finished)
        results = finished
        isProcessing = false
        progress = "Done — \(finished.count) photo\(finished.count == 1 ? "" : "s")"
    }
}

struct GalleryView: View {
    @StateObject private var vm = GalleryViewModel()
    @State private var pickerItems: [PhotosPickerItem] = []
    @State private var showSettings = false
    @AppStorage("speciesModel") private var speciesModelRaw: String = SpeciesModel.subset98.rawValue

    private var speciesModel: SpeciesModel {
        SpeciesModel(rawValue: speciesModelRaw) ?? .subset98
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    header
                    if let err = vm.errorMessage {
                        Text(err).foregroundStyle(.red).multilineTextAlignment(.center)
                    }
                    LazyVStack(spacing: 12) {
                        ForEach(vm.results) { result in
                            ResultCardView(result: result)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("BirdBrained")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showSettings = true
                    } label: {
                        Image(systemName: "gearshape")
                    }
                }
            }
            .sheet(isPresented: $showSettings) {
                SettingsView()
            }
        }
    }

    private var header: some View {
        VStack(spacing: 12) {
            PhotosPicker(selection: $pickerItems,
                         maxSelectionCount: 50,
                         matching: .images) {
                Label("Pick photos", systemImage: "photo.stack")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.accentColor.opacity(0.12))
                    .foregroundStyle(Color.accentColor)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            }
            .disabled(vm.isProcessing)
            .onChange(of: pickerItems) { _, items in
                guard !items.isEmpty else { return }
                let selected = items
                Task {
                    await vm.process(selection: selected, model: speciesModel)
                    pickerItems = []
                }
            }

            HStack {
                Text("Model: \(speciesModel.displayName)")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                if vm.isProcessing {
                    ProgressView().controlSize(.small)
                }
            }

            if !vm.progress.isEmpty {
                Text(vm.progress)
                    .font(.caption).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
}

#Preview {
    GalleryView()
}
