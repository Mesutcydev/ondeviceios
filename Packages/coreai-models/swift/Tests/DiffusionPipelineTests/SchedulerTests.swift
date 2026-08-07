// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAIDiffusionPipeline

@Suite("RNG Sources")
struct RNGTests {
    // Reference values generated with: numpy.random.RandomState(42).standard_normal() × 4
    @Test("NumPy RNG matches Python reference (seed=42)")
    func numpyReference() {
        var rng = NumPyRandomSource(seed: 42)
        let samples = (0..<4).map { _ in rng.nextNormal(mean: 0, stdev: 1) }

        #expect(abs(samples[0] - 0.4967141530) < 1e-6)
        #expect(abs(samples[1] - (-0.1382643012)) < 1e-6)
        #expect(abs(samples[2] - 0.6476885381) < 1e-6)
        #expect(abs(samples[3] - 1.5230298564) < 1e-6)
    }

    @Test("NumPy RNG deterministic across runs")
    func numpyDeterministic() {
        var rng1 = NumPyRandomSource(seed: 123)
        var rng2 = NumPyRandomSource(seed: 123)
        let a = (0..<100).map { _ in rng1.nextNormal(mean: 0, stdev: 1) }
        let b = (0..<100).map { _ in rng2.nextNormal(mean: 0, stdev: 1) }
        #expect(a == b)
    }

    @Test("Torch RNG deterministic across runs")
    func torchDeterministic() {
        var rng1 = TorchRandomSource(seed: 42)
        var rng2 = TorchRandomSource(seed: 42)
        let a = rng1.normalArray([64], mean: 0, stdev: 1)
        let b = rng2.normalArray([64], mean: 0, stdev: 1)
        #expect(a == b)
    }

    @Test("Torch RNG batch-16 path produces expected distribution")
    func torchBatch16() {
        var rng = TorchRandomSource(seed: 7)
        let samples = rng.normalArray([1024], mean: 0, stdev: 1)
        let mean = samples.reduce(0, +) / Float(samples.count)
        let variance = samples.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(samples.count)
        #expect(abs(mean) < 0.1)
        #expect(abs(variance - 1.0) < 0.15)
    }

    @Test("Nv RNG deterministic across runs")
    func nvDeterministic() {
        var rng1 = NvRandomSource(seed: 42)
        var rng2 = NvRandomSource(seed: 42)
        let a = rng1.normalArray([256], mean: 0, stdev: 1)
        let b = rng2.normalArray([256], mean: 0, stdev: 1)
        #expect(a == b)
    }

    @Test("Different seeds produce different sequences")
    func differentSeeds() {
        var rng1 = TorchRandomSource(seed: 1)
        var rng2 = TorchRandomSource(seed: 2)
        let a = rng1.normalArray([16], mean: 0, stdev: 1)
        let b = rng2.normalArray([16], mean: 0, stdev: 1)
        #expect(a != b)
    }

    @Test("normalArray respects mean and stdev")
    func meanStdev() {
        var rng = TorchRandomSource(seed: 42)
        let samples = rng.normalArray([4096], mean: 5.0, stdev: 2.0)
        let mean = samples.reduce(0, +) / Float(samples.count)
        #expect(abs(mean - 5.0) < 0.2)
    }
}

@Suite("Schedulers")
struct SchedulerTests {
    @Test("PNDM timesteps are decreasing")
    func pndmTimesteps() {
        let scheduler = PNDMScheduler(stepCount: 20)
        let ts = scheduler.timeSteps
        #expect(ts.count == 21)  // stepCount - 1 + 2 extra
        #expect(ts.first! > ts.last!)
    }

    @Test("PNDM step produces output of same size as input")
    func pndmStepShape() {
        let scheduler = PNDMScheduler(stepCount: 20)
        let sample = [Float](repeating: 1.0, count: 64)
        let noise = [Float](repeating: 0.5, count: 64)
        let result = scheduler.step(output: noise, timeStep: scheduler.timeSteps[0], sample: sample)
        #expect(result.count == 64)
    }

    @Test("PNDM multiple steps reduce noise")
    func pndmConverges() {
        let scheduler = PNDMScheduler(stepCount: 20)
        var sample = [Float](repeating: 1.0, count: 16)
        for t in scheduler.timeSteps {
            let noise = [Float](repeating: 0.01, count: 16)
            sample = scheduler.step(output: noise, timeStep: t, sample: sample)
        }
        let magnitude = sample.map { abs($0) }.reduce(0, +) / Float(sample.count)
        #expect(magnitude < 100)
    }

    @Test("DPM-Solver++ timesteps are decreasing")
    func dpmTimesteps() {
        let scheduler = DPMSolverMultistepScheduler(stepCount: 20)
        let ts = scheduler.timeSteps
        #expect(ts.first! > ts.last!)
    }

    @Test("DPM-Solver++ step produces output of same size")
    func dpmStepShape() {
        let scheduler = DPMSolverMultistepScheduler(stepCount: 20)
        let sample = [Float](repeating: 1.0, count: 64)
        let noise = [Float](repeating: 0.5, count: 64)
        let result = scheduler.step(output: noise, timeStep: scheduler.timeSteps[0], sample: sample)
        #expect(result.count == 64)
    }

    @Test("DiscreteFlow timesteps constructed correctly")
    func flowTimesteps() {
        let scheduler = DiscreteFlowScheduler(stepCount: 28, trainStepCount: 1000, timeStepShift: 3.0)
        let ts = scheduler.timeSteps
        #expect(ts.first! > ts.last!)
        #expect(ts.count == 28)
    }

    @Test("DiscreteFlow step produces output of same size")
    func flowStepShape() {
        let scheduler = DiscreteFlowScheduler(stepCount: 28)
        let sample = [Float](repeating: 1.0, count: 64)
        let noise = [Float](repeating: 0.5, count: 64)
        let result = scheduler.step(output: noise, timeStep: scheduler.timeSteps[0], sample: sample)
        #expect(result.count == 64)
    }

    @Test("DiscreteFlow shift=1 produces linear sigma schedule")
    func flowLinearSigma() {
        let scheduler = DiscreteFlowScheduler(stepCount: 10, trainStepCount: 1000, timeStepShift: 1.0)
        let firstSigma = scheduler.sigmas.first!
        let lastSigma = scheduler.sigmas.last!
        #expect(firstSigma > lastSigma)
        #expect(abs(lastSigma) < 0.2)
    }

    @Test("PNDM calculateTimesteps with strength")
    func pndmStrength() {
        let scheduler = PNDMScheduler(stepCount: 20)
        let full = scheduler.calculateTimesteps(strength: nil)
        let half = scheduler.calculateTimesteps(strength: 0.5)
        #expect(half.count < full.count)
        #expect(half.count == full.count - 10)
    }

    @Test("linspace produces correct endpoints")
    func linspaceEndpoints() {
        let result = linspace(0.0, 1.0, 11)
        #expect(result.count == 11)
        #expect(abs(result.first! - 0.0) < 1e-6)
        #expect(abs(result.last! - 1.0) < 1e-6)
        #expect(abs(result[5] - 0.5) < 1e-6)
    }

    @Test("weightedSum is correct")
    func weightedSumCorrect() {
        let a: [Float] = [1, 2, 3]
        let b: [Float] = [4, 5, 6]
        let result = weightedSum([0.5, 0.5], [a, b])
        #expect(abs(result[0] - 2.5) < 1e-6)
        #expect(abs(result[1] - 3.5) < 1e-6)
        #expect(abs(result[2] - 4.5) < 1e-6)
    }

    @Test("addNoise blends sample and noise correctly at boundary and midpoint strengths")
    func addNoiseBehavior() {
        let scheduler = DiscreteFlowScheduler(stepCount: 20)
        let sample: [Float] = [1, 2, 3, 4]
        let noise: [Float] = [9, 8, 7, 6]

        // strength=0: original sample unchanged
        #expect(scheduler.addNoise(to: sample, noise: noise, at: 0.0) == sample)
        // strength=1: pure noise
        #expect(scheduler.addNoise(to: sample, noise: noise, at: 1.0) == noise)
        // strength=0.5: exact midpoint
        let mid = scheduler.addNoise(to: [0, 0], noise: [2, 4], at: 0.5)
        #expect(abs(mid[0] - 1.0) < 1e-6)
        #expect(abs(mid[1] - 2.0) < 1e-6)
    }

    @Test("DiscreteFlow sigmaMax constrains schedule start for img2img")
    func sigmaMaxSchedule() {
        let strength: Float = 0.85
        let scheduler = DiscreteFlowScheduler(
            stepCount: 20, trainStepCount: 1000, timeStepShift: 1.0, sigmaMax: strength)
        #expect(scheduler.sigmas.first! <= strength + 1e-5)
        #expect(scheduler.startSigma == scheduler.sigmas.first!)

        // sigmaMax=1.0 matches the default (unconstrained) schedule
        let def = DiscreteFlowScheduler(stepCount: 20, trainStepCount: 1000, timeStepShift: 1.0)
        let withMax = DiscreteFlowScheduler(
            stepCount: 20, trainStepCount: 1000, timeStepShift: 1.0, sigmaMax: 1.0)
        #expect(def.sigmas == withMax.sigmas)
    }
}
