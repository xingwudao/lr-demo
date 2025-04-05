declare module 'ml-logistic-regression' {
  interface LogisticRegressionOptions {
    numSteps?: number
    learningRate?: number
  }

  class LogisticRegression {
    constructor(options?: LogisticRegressionOptions)
    train(X: number[][], y: number[]): void
    predict(X: number[][]): number[]
    loss(X: number[][], y: number[]): number
    weights: number[]
  }

  export = LogisticRegression
} 