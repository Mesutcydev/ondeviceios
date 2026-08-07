import XCTest
@testable import IOSLocalLLM

final class WebSearchServiceTests: XCTestCase {
    func test_defaultProviderSupportsQuerySearch() {
        XCTAssertEqual(WebToolSettings().provider, .duckduckgoHTML)
    }

    func test_duckDuckGoParser_handlesWrappedResultURLAndEntities() {
        let html = """
        <html><body>
          <a class='result__a' href='/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs%3Fa%3D1%26b%3D2'>Example &amp; Docs</a>
          <a class='result__snippet'>A &lt;great&gt; snippet</a>
        </body></html>
        """

        let results = WebSearchService.parseDuckDuckGoHTMLResults(html, max: 5)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].title, "Example & Docs")
        XCTAssertEqual(results[0].url.absoluteString, "https://example.com/docs?a=1&b=2")
        XCTAssertEqual(results[0].snippet, "A <great> snippet")
    }

    func test_duckDuckGoParser_handlesAbsoluteResultURL() {
        let html = """
        <a href="https://swift.org/blog/" class="result__a highlighted">Swift Blog</a>
        """
        let results = WebSearchService.parseDuckDuckGoHTMLResults(html, max: 1)
        XCTAssertEqual(results.first?.url.absoluteString, "https://swift.org/blog/")
    }
}
