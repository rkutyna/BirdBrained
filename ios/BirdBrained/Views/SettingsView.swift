import SwiftUI

struct SettingsView: View {
    @AppStorage("speciesModel") private var speciesModelRaw: String = SpeciesModel.subset98.rawValue
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    ForEach(SpeciesModel.allCases) { model in
                        Button {
                            speciesModelRaw = model.rawValue
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(model.displayName).foregroundStyle(.primary)
                                    Text("\(model.classCount) classes")
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                                Spacer()
                                if speciesModelRaw == model.rawValue {
                                    Image(systemName: "checkmark").foregroundStyle(Color.accentColor)
                                }
                            }
                        }
                    }
                } header: {
                    Text("Species model")
                } footer: {
                    Text("Switch between the curated 98-species model (higher accuracy on common birds) and the full 404-species NABirds model.")
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

#Preview {
    SettingsView()
}
