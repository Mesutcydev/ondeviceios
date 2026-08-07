import Foundation
import os

/// One admission point for inference initiated by any network server.
/// Local UI generation remains authoritative; remote requests fail fast while
/// the model is already generating instead of queueing surprising work.
final class RemoteInferenceGate: @unchecked Sendable {
    static let shared = RemoteInferenceGate()

    private let owner = OSAllocatedUnfairLock<UUID?>(initialState: nil)

    func acquire() -> UUID? {
        owner.withLock { currentOwner in
            guard currentOwner == nil else { return nil }
            let token = UUID()
            currentOwner = token
            return token
        }
    }

    func release(_ token: UUID) {
        owner.withLock { currentOwner in
            if currentOwner == token {
                currentOwner = nil
            }
        }
    }
}
