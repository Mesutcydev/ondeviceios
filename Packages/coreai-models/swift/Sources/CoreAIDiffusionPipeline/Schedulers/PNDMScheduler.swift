// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import Foundation

/// How to map a beta range to a sequence of betas.
public enum BetaSchedule {
    case linear
    case scaledLinear
}

/// What the model predicts at each denoising step.
public enum PredictionType: String, Codable, Sendable {
    case epsilon
    case vPrediction = "v_prediction"
    case flow
    case flowMatching = "flow_matching"
}

/// PNDM (Pseudo Numerical Methods for Diffusion Models) scheduler.
/// Matches HuggingFace Diffusers PNDMScheduler (PLMS method only).
public final class PNDMScheduler {
    public let trainStepCount: Int
    public let inferenceStepCount: Int
    public let betas: [Float]
    public let alphas: [Float]
    public let alphasCumProd: [Float]
    public let timeSteps: [Int]

    let alphaT: [Float]
    let sigmaT: [Float]

    public let predictionType: PredictionType

    public private(set) var modelOutputs: [[Float]] = []

    var counter: Int
    var ets: [[Float]]
    var currentSample: [Float]?

    public init(
        stepCount: Int = 50,
        trainStepCount: Int = 1000,
        betaSchedule: BetaSchedule = .scaledLinear,
        betaStart: Float = 0.00085,
        betaEnd: Float = 0.012,
        predictionType: PredictionType = .epsilon
    ) {
        self.trainStepCount = trainStepCount
        self.inferenceStepCount = stepCount
        self.predictionType = predictionType

        switch betaSchedule {
        case .linear:
            self.betas = linspace(betaStart, betaEnd, trainStepCount)
        case .scaledLinear:
            self.betas = linspace(pow(betaStart, 0.5), pow(betaEnd, 0.5), trainStepCount).map { $0 * $0 }
        }
        self.alphas = betas.map { 1.0 - $0 }
        var cumProd = self.alphas
        for i in 1..<cumProd.count {
            cumProd[i] *= cumProd[i - 1]
        }
        self.alphasCumProd = cumProd

        let stepsOffset = 1
        let stepRatio = Float(trainStepCount / stepCount)
        let forwardSteps = (0..<stepCount).map {
            Int((Float($0) * stepRatio).rounded()) + stepsOffset
        }

        self.alphaT = vForce.sqrt(self.alphasCumProd)
        self.sigmaT = vForce.sqrt(
            vDSP.subtract([Float](repeating: 1, count: self.alphasCumProd.count), self.alphasCumProd))

        var ts: [Int] = []
        ts.append(contentsOf: forwardSteps.dropLast(1))
        ts.append(ts.last!)
        ts.append(forwardSteps.last!)
        ts.reverse()
        self.timeSteps = ts

        self.counter = 0
        self.ets = []
        self.currentSample = nil
    }

    public func step(output: [Float], timeStep t: Int, sample s: [Float]) -> [Float] {
        var timeStep = t
        let stepInc = trainStepCount / inferenceStepCount
        var prevStep = timeStep - stepInc
        var modelOutput = output
        var sample = s

        if counter != 1 {
            if ets.count > 3 {
                ets = Array(ets[(ets.count - 3)...])
            }
            ets.append(output)
        } else {
            prevStep = timeStep
            timeStep = timeStep + stepInc
        }

        if ets.count == 1 && counter == 0 {
            modelOutput = output
            currentSample = sample
        } else if ets.count == 1 && counter == 1 {
            modelOutput = weightedSum([0.5, 0.5], [output, ets[ets.count - 1]])
            sample = currentSample!
            currentSample = nil
        } else if ets.count == 2 {
            modelOutput = weightedSum([1.5, -0.5], [ets[ets.count - 1], ets[ets.count - 2]])
        } else if ets.count == 3 {
            modelOutput = weightedSum(
                [23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0],
                [ets[ets.count - 1], ets[ets.count - 2], ets[ets.count - 3]])
        } else {
            modelOutput = weightedSum(
                [55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0],
                [ets[ets.count - 1], ets[ets.count - 2], ets[ets.count - 3], ets[ets.count - 4]])
        }

        let convertedOutput = convertModelOutput(modelOutput: modelOutput, timestep: timeStep, sample: sample)
        modelOutputs.append(convertedOutput)

        let prevSample = previousSample(sample, timeStep, prevStep, modelOutput)
        counter += 1
        return prevSample
    }

    func convertModelOutput(modelOutput: [Float], timestep: Int, sample: [Float]) -> [Float] {
        let count = modelOutput.count
        let (at, st) = (alphaT[timestep], sigmaT[timestep])
        var result = [Float](repeating: 0, count: count)
        switch predictionType {
        case .epsilon:
            for i in 0..<count {
                result[i] = (sample[i] - modelOutput[i] * st) / at
            }
        case .vPrediction:
            for i in 0..<count {
                result[i] = at * sample[i] - st * modelOutput[i]
            }
        case .flow:
            preconditionFailure("PNDMScheduler does not support flow prediction; use DiscreteFlowScheduler")
        case .flowMatching:
            fatalError("PNDMScheduler does not support flowMatching prediction type")
        }
        return result
    }

    func previousSample(_ sample: [Float], _ timeStep: Int, _ prevStep: Int, _ modelOutput: [Float]) -> [Float] {
        let clampedStep = min(timeStep, alphasCumProd.count - 1)
        let clampedPrev = min(max(0, prevStep), alphasCumProd.count - 1)
        let alphaProdt = alphasCumProd[clampedStep]
        let alphaProdtPrev = alphasCumProd[clampedPrev]
        let betaProdt = 1 - alphaProdt
        let betaProdtPrev = 1 - alphaProdtPrev

        let sampleCoeff = sqrt(alphaProdtPrev / alphaProdt)
        let modelOutputDenomCoeff = alphaProdt * sqrt(betaProdtPrev) + sqrt(alphaProdt * betaProdt * alphaProdtPrev)
        let modelCoeff = -(alphaProdtPrev - alphaProdt) / modelOutputDenomCoeff

        return weightedSum([sampleCoeff, modelCoeff], [sample, modelOutput])
    }

    public func calculateTimesteps(strength: Float?) -> [Int] {
        guard let strength else { return timeSteps }
        let startStep = max(inferenceStepCount - Int(Float(inferenceStepCount) * strength), 0)
        return Array(timeSteps[startStep...])
    }

    public func addNoise(originalSample: [Float], noise: [[Float]], strength: Float) -> [[Float]] {
        let startStep = max(inferenceStepCount - Int(Float(inferenceStepCount) * strength), 0)
        let alphaProdt = alphasCumProd[timeSteps[startStep]]
        let sqrtAlpha = sqrt(alphaProdt)
        let sqrtBeta = sqrt(1 - alphaProdt)
        return noise.map { weightedSum([sqrtAlpha, sqrtBeta], [originalSample, $0]) }
    }
}
