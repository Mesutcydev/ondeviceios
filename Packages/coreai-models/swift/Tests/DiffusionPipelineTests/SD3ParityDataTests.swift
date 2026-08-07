// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@Suite("SD3 parity data")
struct SD3ParityDataTests {
    // Parity fixtures are optional — not all workspace checkouts include them.
    // Gate the whole suite on their presence so CI stays green while local
    // devs keep full coverage.
    static let parityDataAvailable: Bool = {
        let probe = parityDir().appendingPathComponent("text_encoder_pooled.npy")
        return FileManager.default.fileExists(atPath: probe.path)
    }()

    // MARK: - Contract

    struct Expect: Sendable {
        let file: String
        let shape: [Int]
        let dtype: Dtype
    }

    enum Dtype: Sendable { case float32, int64 }

    private static let contract: [Expect] = [
        .init(file: "text_encoder_input_ids.npy", shape: [1, 77], dtype: .int64),
        .init(file: "text_encoder_hidden.npy", shape: [1, 77, 768], dtype: .float32),
        .init(file: "text_encoder_pooled.npy", shape: [1, 768], dtype: .float32),
        .init(file: "text_encoder_2_input_ids.npy", shape: [1, 77], dtype: .int64),
        .init(file: "text_encoder_2_hidden.npy", shape: [1, 77, 1280], dtype: .float32),
        .init(file: "text_encoder_2_pooled.npy", shape: [1, 1280], dtype: .float32),
        .init(file: "mmdit_hidden_states.npy", shape: [2, 16, 128, 128], dtype: .float32),
        .init(file: "mmdit_timestep.npy", shape: [2], dtype: .float32),
        .init(file: "mmdit_encoder_hidden_states.npy", shape: [2, 154, 4096], dtype: .float32),
        .init(file: "mmdit_pooled_projections.npy", shape: [2, 2048], dtype: .float32),
        .init(file: "mmdit_output.npy", shape: [2, 16, 128, 128], dtype: .float32),
        .init(file: "vae_decoder_input.npy", shape: [1, 16, 128, 128], dtype: .float32),
        .init(file: "vae_decoder_output.npy", shape: [1, 3, 1024, 1024], dtype: .float32),
    ]

    // MARK: - Tests

    @Test(
        "All 13 reference tensors load with the expected shape and dtype",
        .enabled(if: Self.parityDataAvailable),
        arguments: Self.contract)
    func loadTensor(_ expected: Expect) throws {
        let url = Self.parityDir().appendingPathComponent(expected.file)
        let header = try NpyHeader.parse(contentsOf: url)

        #expect(header.shape == expected.shape, "shape mismatch for \(expected.file)")
        switch expected.dtype {
        case .float32:
            #expect(
                header.dtypeToken == "f4" || header.dtypeToken == "<f4",
                "dtype mismatch for \(expected.file): got \(header.dtypeToken)")
        case .int64:
            #expect(
                header.dtypeToken == "i8" || header.dtypeToken == "<i8",
                "dtype mismatch for \(expected.file): got \(header.dtypeToken)")
        }

        let expectedBytes = header.shape.reduce(1, *) * (expected.dtype == .int64 ? 8 : 4)
        #expect(
            header.payloadByteCount == expectedBytes,
            "payload size mismatch for \(expected.file)")
    }

    @Test(
        "MMDiT input channel dim matches joint_attention_dim derived from SD 3.5 Medium config",
        .enabled(if: Self.parityDataAvailable))
    func mmditChannelDim() throws {
        let url = Self.parityDir().appendingPathComponent("mmdit_encoder_hidden_states.npy")
        let header = try NpyHeader.parse(contentsOf: url)
        #expect(header.shape.last == 4096)
    }

    @Test(
        "Pooled projection dim = CLIP-L pooled + CLIP-G pooled",
        .enabled(if: Self.parityDataAvailable))
    func pooledProjectionDim() throws {
        let clipL = try NpyHeader.parse(contentsOf: Self.parityDir().appendingPathComponent("text_encoder_pooled.npy"))
        let clipG = try NpyHeader.parse(
            contentsOf: Self.parityDir().appendingPathComponent("text_encoder_2_pooled.npy"))
        let mmdit = try NpyHeader.parse(
            contentsOf: Self.parityDir().appendingPathComponent("mmdit_pooled_projections.npy"))
        #expect((clipL.shape.last ?? 0) + (clipG.shape.last ?? 0) == (mmdit.shape.last ?? 0))
    }

    // MARK: - Helpers

    private static func parityDir() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // DiffusionPipeline_Tests/
            .deletingLastPathComponent()  // Tests/
            .deletingLastPathComponent()  // swift/
            .deletingLastPathComponent()  // repo root
            .appendingPathComponent("internal/validation/sd3-medium/parity", isDirectory: true)
    }
}

// MARK: - Minimal .npy header parser (shape + dtype only, no payload load)

private struct NpyHeader {
    let shape: [Int]
    let dtypeToken: String
    let payloadByteCount: Int

    static func parse(contentsOf url: URL) throws -> NpyHeader {
        let raw = try Data(contentsOf: url)
        guard raw.count > 10, raw[0] == 0x93,
            raw[1] == UInt8(ascii: "N"), raw[2] == UInt8(ascii: "U"), raw[3] == UInt8(ascii: "M"),
            raw[4] == UInt8(ascii: "P"), raw[5] == UInt8(ascii: "Y")
        else {
            throw NSError(
                domain: "NpyHeader", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "bad magic in \(url.lastPathComponent)"])
        }
        let version = raw[6]
        let headerLen: Int
        let headerStart: Int
        if version == 1 {
            headerLen = Int(raw[8]) | (Int(raw[9]) << 8)
            headerStart = 10
        } else {
            headerLen =
                Int(raw[8]) | (Int(raw[9]) << 8)
                | (Int(raw[10]) << 16) | (Int(raw[11]) << 24)
            headerStart = 12
        }
        let dataStart = headerStart + headerLen
        let header = String(data: raw[headerStart..<dataStart], encoding: .ascii) ?? ""

        return NpyHeader(
            shape: Self.parseShape(header),
            dtypeToken: Self.parseDtype(header),
            payloadByteCount: raw.count - dataStart
        )
    }

    private static func parseShape(_ header: String) -> [Int] {
        guard let start = header.range(of: "("),
            let end = header.range(of: ")", range: start.upperBound..<header.endIndex)
        else { return [] }
        return header[start.upperBound..<end.lowerBound]
            .split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    }

    private static func parseDtype(_ header: String) -> String {
        // Header looks like: "'descr': '<f4', ..."
        guard let descrRange = header.range(of: "'descr':") else { return "" }
        let after = header[descrRange.upperBound...]
        guard let open = after.firstIndex(of: "'"),
            let close = after[after.index(after: open)...].firstIndex(of: "'")
        else { return "" }
        return String(after[after.index(after: open)..<close])
    }
}

extension SD3ParityDataTests.Expect: CustomTestStringConvertible {
    var testDescription: String { file }
}
