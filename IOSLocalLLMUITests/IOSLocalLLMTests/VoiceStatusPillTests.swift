import XCTest
@testable import IOSLocalLLM

// MARK: - VoiceStatusPillTests
//
// The pill itself is a SwiftUI view (hard to assert against without
// snapshot tests, out of scope). What we CAN pin is the BCP-47 →
// flag-emoji conversion, which is the only piece of logic the view
// contains and the bit a user notices most when it's wrong.

final class VoiceStatusPillTests: XCTestCase {

    func test_flag_enUS() {
        XCTAssertEqual(VoiceStatusPill.flagEmoji(forBCP47: "en-US"), "🇺🇸")
    }

    func test_flag_trTR() {
        XCTAssertEqual(VoiceStatusPill.flagEmoji(forBCP47: "tr-TR"), "🇹🇷")
    }

    func test_flag_deDE() {
        XCTAssertEqual(VoiceStatusPill.flagEmoji(forBCP47: "de-DE"), "🇩🇪")
    }

    func test_flag_lowercaseRegionStillResolves() {
        // The mapping uppercases internally — `en-us` should produce
        // the same flag as `en-US`.
        XCTAssertEqual(VoiceStatusPill.flagEmoji(forBCP47: "en-us"), "🇺🇸")
    }

    func test_flag_bareLanguageReturnsNil() {
        // No region — pill hides the flag slot rather than showing
        // a fallback glyph that would mislead the user.
        XCTAssertNil(VoiceStatusPill.flagEmoji(forBCP47: "en"))
    }

    func test_flag_nilInput_returnsNil() {
        XCTAssertNil(VoiceStatusPill.flagEmoji(forBCP47: nil))
    }

    func test_flag_emptyString_returnsNil() {
        XCTAssertNil(VoiceStatusPill.flagEmoji(forBCP47: ""))
    }

    func test_flag_invalidRegion_returnsNil() {
        // Three-letter "region" is not a real BCP-47 code — return
        // nil rather than guessing.
        XCTAssertNil(VoiceStatusPill.flagEmoji(forBCP47: "en-USA"))
        // Region with digits.
        XCTAssertNil(VoiceStatusPill.flagEmoji(forBCP47: "en-12"))
    }

    // MARK: - Common locale roundtrips that the registry produces

    func test_flag_languagesFromBuiltinProfiles() {
        // The LanguageDetector's BCP-47 mapper folds `en` → `en-US`,
        // `tr` → `tr-TR`, etc. The pill must render flags for the
        // set the built-in profiles' `dominantLanguages` produce.
        XCTAssertEqual(VoiceStatusPill.flagEmoji(forBCP47: "en-US"), "🇺🇸")
        XCTAssertEqual(VoiceStatusPill.flagEmoji(forBCP47: "tr-TR"), "🇹🇷")
        XCTAssertEqual(VoiceStatusPill.flagEmoji(forBCP47: "es-ES"), "🇪🇸")
        XCTAssertEqual(VoiceStatusPill.flagEmoji(forBCP47: "fr-FR"), "🇫🇷")
        XCTAssertEqual(VoiceStatusPill.flagEmoji(forBCP47: "ja-JP"), "🇯🇵")
        XCTAssertEqual(VoiceStatusPill.flagEmoji(forBCP47: "zh-CN"), "🇨🇳")
    }
}
