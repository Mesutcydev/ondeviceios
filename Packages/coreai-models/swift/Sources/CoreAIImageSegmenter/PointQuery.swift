// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics

/// Point (and box) prompt for point-guided segmentation models such as EfficientSAM.
///
/// A `PointQuery` holds **per-query** points: `queries[q]` is the list of points that
/// belong to the *same* prompt — they are fused into one mask by the model. The outer
/// dimension (`queries.count`) is `Q` (the `num_queries` dim of `batched_points`); the
/// inner dimension (`queries[q].count`) is `P` (the `num_pts` dim).
///
/// Common shapes:
/// - **Single click** — `Q=1, P=1`: one query with one foreground point.
/// - **Box prompt** — `Q=1, P=2`: one query with `[.boxTopLeft, .boxBottomRight]`.
/// - **Click + negative click** — `Q=1, P=2`: one query with `[.foreground, .background]`.
/// - **Independent prompts** — `Q=N, P=1`: N queries each with one point.
/// - **Box per query** — `Q=N, P=2`: N queries each with a box.
///
/// Each `Point` carries pixel coordinates relative to the input image and a `Label` that
/// tells the model whether the point is a foreground/background click or a box corner:
///
/// - `.foreground` (1) — the model should include this location in the mask.
/// - `.background` (0) — the model should exclude this location from the mask.
/// - `.boxTopLeft` (2) / `.boxBottomRight` (3) — together define a bounding-box prompt.
///
/// Coordinates are in **input-image pixel space**: `x` in `[0, imageWidth]`,
/// `y` in `[0, imageHeight]`. The engine scales them to the model's internal resolution.
///
/// When `queries` is empty the engine falls back to a `gridSide × gridSide` grid of
/// foreground points (one point per query), which is equivalent to "segment everything."
///
/// The engine validates `queries.count` and each `queries[q].count` against the model's
/// static `batched_points` shape and pads missing slots with sentinel values so any
/// `(Q, P)` smaller than the model's accepted shape is accepted.
public struct PointQuery: Sendable {
    /// Role of a single point in the prompt.
    public enum Label: Int32, Sendable {
        case background = 0
        case foreground = 1
        /// Top-left corner of a bounding-box prompt. Pair with `.boxBottomRight`.
        case boxTopLeft = 2
        /// Bottom-right corner of a bounding-box prompt. Pair with `.boxTopLeft`.
        case boxBottomRight = 3
    }

    /// A single prompt point in input-image pixel coordinates.
    public struct Point: Sendable {
        /// Horizontal position in input-image pixels (`[0, imageWidth]`).
        public var x: Float
        /// Vertical position in input-image pixels (`[0, imageHeight]`).
        public var y: Float
        /// Role of this point (foreground click, background click, or box corner).
        public var label: Label

        public init(x: Float, y: Float, label: Label = .foreground) {
            self.x = x
            self.y = y
            self.label = label
        }
    }

    /// Prompt points grouped by query. Outer = queries (`Q`), inner = points per query (`P`).
    /// Empty means "segment everything" — the engine substitutes a `gridSide × gridSide`
    /// grid of foreground points where `gridSide = sqrt(num_queries)`.
    public var queries: [[Point]]

    /// Build from an explicit per-query layout.
    public init(queries: [[Point]] = []) {
        self.queries = queries
    }

    /// Build a single-query prompt that fuses `points` into one mask.
    ///
    /// Pass `[tl, br]` with labels `[.boxTopLeft, .boxBottomRight]` for a box prompt, or a
    /// mix of foreground/background clicks targeting the same object. Use `init(queries:)`
    /// when you need multiple independent prompts. To trigger segment-everything mode use
    /// the no-arg `PointQuery()` initializer.
    public init(points: [Point]) {
        self.queries = [points]
    }
}
