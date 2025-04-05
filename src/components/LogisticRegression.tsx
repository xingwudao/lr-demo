import { useState, useEffect, useRef } from 'react'
import * as d3 from 'd3'
import { Matrix } from 'ml-matrix'
import './LogisticRegression.css'

interface DataPoint {
  studyHours: number
  attendance: number
  passed: number
}

interface TrainingPoint {
  iteration: number
  loss: number
}

// 生成随机权重，范围在 [-0.1, 0.1] 之间
const generateRandomWeights = () => {
  return Array(3).fill(0).map(() => (Math.random() - 0.5) * 0.2)
}

const LogisticRegressionDemo = () => {
  const trainingRef = useRef<SVGSVGElement>(null)
  const predictionRef = useRef<SVGSVGElement>(null)
  const [data, setData] = useState<DataPoint[]>([])
  const [normalizedData, setNormalizedData] = useState<DataPoint[]>([])
  const [normalizedParams, setNormalizedParams] = useState<{ studyHours: { mean: number; std: number }; attendance: { mean: number; std: number } }>({ studyHours: { mean: 0, std: 1 }, attendance: { mean: 0, std: 1 } })
  const [iterations, setIterations] = useState(200)
  const [isTraining, setIsTraining] = useState(false)
  const [currentIteration, setCurrentIteration] = useState(0)
  const [theta, setTheta] = useState<number[]>(() => generateRandomWeights())
  const [trainingHistory, setTrainingHistory] = useState<{ iteration: number; loss: number }[]>([])
  const [inputPoint, setInputPoint] = useState({ studyHours: 0, attendance: 0 })
  const [prediction, setPrediction] = useState(0)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // 计算sigmoid函数
  const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))

  // 计算交叉熵损失
  const computeLoss = (X: Matrix, y: Matrix, theta: Matrix) => {
    const m = X.rows
    const h = X.mmul(theta).to2DArray().map(row => sigmoid(row[0]))
    const yArray = y.to2DArray().map(row => row[0])
    let loss = 0
    for (let i = 0; i < m; i++) {
      loss += -yArray[i] * Math.log(h[i]) - (1 - yArray[i]) * Math.log(1 - h[i])
    }
    return loss / m
  }

  // 实现Logistic Regression
  const logisticRegression = (X: Matrix, y: Matrix, currentTheta: Matrix) => {
    const m = X.rows
    const alpha = 0.3 // 增加学习率

    // 计算预测值
    const h = X.mmul(currentTheta).to2DArray().map(row => sigmoid(row[0]))

    // 计算误差
    const errors = h.map((hi, i) => hi - y.to2DArray()[i][0])

    // 计算梯度
    const gradient = new Matrix(currentTheta.rows, 1)
    for (let j = 0; j < currentTheta.rows; j++) {
      let sum = 0
      for (let i = 0; i < m; i++) {
        sum += errors[i] * X.get(i, j)
      }
      gradient.set(j, 0, (alpha * sum) / m)
    }

    // 更新参数
    return new Matrix(currentTheta.to2DArray().map((row, i) => [row[0] - gradient.get(i, 0)]))
  }

  // 加载数据
  useEffect(() => {
    const loadData = async () => {
      try {
        setIsLoading(true)
        setError(null)
        const response = await fetch('/data.csv')
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }
        const text = await response.text()
        const rows = text.split('\n').slice(1) // 跳过标题行
        const dataPoints = rows
          .filter(row => row.trim()) // 移除空行
          .map(row => {
            const [studyHours, attendance, passed] = row.split(',').map(Number)
            return {
              studyHours,
              attendance,
              passed
            }
          })

        // 计算标准化参数
        const studyHoursMean = dataPoints.reduce((sum, p) => sum + p.studyHours, 0) / dataPoints.length
        const studyHoursStd = Math.sqrt(dataPoints.reduce((sum, p) => sum + Math.pow(p.studyHours - studyHoursMean, 2), 0) / dataPoints.length)
        
        const attendanceMean = dataPoints.reduce((sum, p) => sum + p.attendance, 0) / dataPoints.length
        const attendanceStd = Math.sqrt(dataPoints.reduce((sum, p) => sum + Math.pow(p.attendance - attendanceMean, 2), 0) / dataPoints.length)

        // 保存标准化参数
        const normalizedParams = {
          studyHours: { mean: studyHoursMean, std: studyHoursStd },
          attendance: { mean: attendanceMean, std: attendanceStd }
        }

        // 标准化数据用于训练
        const normalizedData = dataPoints.map(p => ({
          studyHours: (p.studyHours - studyHoursMean) / studyHoursStd,
          attendance: (p.attendance - attendanceMean) / attendanceStd,
          passed: p.passed
        }))

        setData(dataPoints) // 保存原始数据用于显示
        setNormalizedData(normalizedData) // 保存标准化数据用于训练
        setNormalizedParams(normalizedParams) // 保存标准化参数
      } catch (error) {
        console.error('加载数据失败:', error)
        setError('加载数据失败，请刷新页面重试')
      } finally {
        setIsLoading(false)
      }
    }

    loadData()
  }, [])

  // 训练模型
  useEffect(() => {
    if (isTraining && currentIteration < iterations) {
      const trainStep = async () => {
        try {
          // 准备数据（使用标准化数据）
          const X = new Matrix(normalizedData.map(p => [1, p.studyHours, p.attendance]))
          const y = new Matrix(normalizedData.map(p => [p.passed]))
          const currentTheta = new Matrix(theta.map(t => [t]))

          // 执行一次梯度下降
          const newTheta = logisticRegression(X, y, currentTheta)
          const loss = computeLoss(X, y, newTheta)

          // 更新状态
          const newThetaArray = newTheta.getColumn(0)
          setTheta(newThetaArray)
          
          // 每10次迭代记录一次损失值，或者在最后一次迭代时记录
          if (currentIteration % 10 === 0 || currentIteration === iterations - 1) {
            setTrainingHistory(prev => [...prev, { iteration: currentIteration, loss }])
          }
          
          setCurrentIteration(prev => prev + 1)
        } catch (error) {
          console.error('训练失败:', error)
          setIsTraining(false)
        }
      }

      // 添加延迟
      const timer = setTimeout(trainStep, 100)
      return () => clearTimeout(timer)
    } else if (currentIteration >= iterations) {
      setIsTraining(false)
    }
  }, [isTraining, currentIteration, normalizedData])

  // 绘制训练曲线
  useEffect(() => {
    if (!trainingRef.current) return

    const svg = d3.select(trainingRef.current)
    const width = 600
    const height = 300
    const margin = { top: 20, right: 20, bottom: 30, left: 50 }

    // 清除之前的图形
    svg.selectAll('*').remove()

    // 创建比例尺
    const xScale = d3.scaleLinear()
      .domain([0, iterations])
      .range([margin.left, width - margin.right])

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(trainingHistory, d => d.loss) || 1])
      .range([height - margin.bottom, margin.top])

    // 绘制坐标轴
    const xAxis = d3.axisBottom(xScale)
      .ticks(10)
      .tickFormat(d => d.toString())

    const yAxis = d3.axisLeft(yScale)

    svg.append('g')
      .attr('class', 'axis')
      .attr('transform', `translate(0, ${height - margin.bottom})`)
      .call(xAxis)

    svg.append('g')
      .attr('class', 'axis')
      .attr('transform', `translate(${margin.left}, 0)`)
      .call(yAxis)

    // 添加坐标轴标签
    svg.append('text')
      .attr('class', 'axis-label')
      .attr('x', width / 2)
      .attr('y', height - 5)
      .style('text-anchor', 'middle')
      .text('迭代次数')

    svg.append('text')
      .attr('class', 'axis-label')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', 15)
      .style('text-anchor', 'middle')
      .text('损失值')

    // 绘制损失曲线
    const line = d3.line<TrainingPoint>()
      .x(d => xScale(d.iteration))
      .y(d => yScale(d.loss))

    svg.append('path')
      .datum(trainingHistory)
      .attr('class', 'loss-curve')
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', '#4CAF50')
      .attr('stroke-width', 2)
  }, [trainingHistory, iterations])

  // 绘制预测区域
  useEffect(() => {
    if (!predictionRef.current) return

    const svg = d3.select(predictionRef.current)
    const width = 400
    const height = 300
    const margin = { top: 20, right: 20, bottom: 30, left: 50 }

    // 清除之前的图形
    svg.selectAll('*').remove()

    // 创建比例尺（使用原始数据范围）
    const xScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.studyHours) || 50])
      .range([margin.left, width - margin.right])

    const yScale = d3.scaleLinear()
      .domain([0, 100])
      .range([height - margin.bottom, margin.top])

    // 绘制坐标轴
    const xAxis = d3.axisBottom(xScale)
    const yAxis = d3.axisLeft(yScale)

    svg.append('g')
      .attr('class', 'axis')
      .attr('transform', `translate(0, ${height - margin.bottom})`)
      .call(xAxis)

    svg.append('g')
      .attr('class', 'axis')
      .attr('transform', `translate(${margin.left}, 0)`)
      .call(yAxis)

    // 添加坐标轴标签
    svg.append('text')
      .attr('class', 'axis-label')
      .attr('x', width / 2)
      .attr('y', height - 5)
      .style('text-anchor', 'middle')
      .text('每周学习时间（小时）')

    svg.append('text')
      .attr('class', 'axis-label')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', 15)
      .style('text-anchor', 'middle')
      .text('出勤率（%）')

    // 绘制数据点（使用原始数据）
    svg.selectAll('.data-point')
      .data(data)
      .enter()
      .append('circle')
      .attr('class', 'data-point')
      .attr('cx', d => xScale(d.studyHours))
      .attr('cy', d => yScale(d.attendance))
      .attr('r', 3)
      .attr('fill', d => d.passed === 1 ? '#4CAF50' : '#F44336')

    // 绘制输入点
    if (inputPoint.studyHours > 0 || inputPoint.attendance > 0) {
      // 标准化输入点用于预测
      const normalizedStudyHours = (inputPoint.studyHours - normalizedParams.studyHours.mean) / normalizedParams.studyHours.std
      const normalizedAttendance = (inputPoint.attendance - normalizedParams.attendance.mean) / normalizedParams.attendance.std

      const z = theta[0] + theta[1] * normalizedStudyHours + theta[2] * normalizedAttendance
      const prob = sigmoid(z)
      setPrediction(prob)

      // 使用原始坐标绘制输入点
      svg.append('circle')
        .attr('class', 'input-point')
        .attr('cx', xScale(inputPoint.studyHours))
        .attr('cy', yScale(inputPoint.attendance))
        .attr('r', 8)
        .attr('fill', prob > 0.5 ? '#4CAF50' : '#F44336')
        .attr('opacity', 0.7)
    }
  }, [data, inputPoint, theta, normalizedParams])

  const startTraining = () => {
    if (isTraining) {
      setIsTraining(false)
      return
    }
    if (currentIteration >= iterations) {
      setTrainingHistory([])
      setCurrentIteration(0)
      setTheta(generateRandomWeights())
    }
    setIsTraining(true)
  }

  if (isLoading) {
    return <div className="loading">加载中...</div>
  }

  if (error) {
    return <div className="error">{error}</div>
  }

  if (data.length === 0) {
    return <div className="error">没有可用的数据</div>
  }

  return (
    <div className="logistic-regression">
      <div className="container">
        <div className="training-section">
          <div className="controls">
            <div className="iteration-control">
              <label>迭代次数：</label>
              <input
                type="number"
                min="1"
                max="10000"
                value={iterations}
                onChange={(e) => setIterations(Number(e.target.value))}
                disabled={isTraining}
              />
            </div>
            <div className="buttons">
              <button onClick={startTraining}>
                {isTraining ? '训练中...' : '开始训练'}
              </button>
            </div>
          </div>
          <div className="training-visualization">
            <svg
              ref={trainingRef}
              width={600}
              height={300}
            />
          </div>
          <div className="weights">
            <h3>当前权重：</h3>
            <p>偏置项：<span>{theta[0].toFixed(4)}</span></p>
            <p>学习时间权重：<span>{theta[1].toFixed(4)}</span></p>
            <p>出勤率权重：<span>{theta[2].toFixed(4)}</span></p>
          </div>
          <div className="formula">
            <h3>预测公式：</h3>
            <p>z = <span>{theta[0].toFixed(4)}</span> + <span>{theta[1].toFixed(4)}</span> × 学习时间 + <span>{theta[2].toFixed(4)}</span> × 出勤率</p>
            <p>通过概率 = 1 ÷ (1 + e的-z次方)</p>
          </div>
        </div>
        <div className="prediction-section">
          <div className="input-controls">
            <div className="input-group">
              <label>每周学习时间（小时）：</label>
              <input
                type="number"
                min="0"
                max="50"
                value={inputPoint.studyHours}
                onChange={(e) => setInputPoint({ ...inputPoint, studyHours: Number(e.target.value) })}
              />
            </div>
            <div className="input-group">
              <label>出勤率（%）：</label>
              <input
                type="number"
                min="0"
                max="100"
                value={inputPoint.attendance}
                onChange={(e) => setInputPoint({ ...inputPoint, attendance: Number(e.target.value) })}
              />
            </div>
          </div>
          <div className="prediction-visualization">
            <svg
              ref={predictionRef}
              width={400}
              height={300}
            />
          </div>
          <div className="prediction-result">
            <h3>预测结果：</h3>
            <p>通过概率：<span>{(prediction * 100).toFixed(2)}%</span></p>
            <p>预测结果：{prediction > 0.5 ? '通过' : '不通过'}</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default LogisticRegressionDemo 