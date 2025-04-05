import LogisticRegressionDemo from './components/LogisticRegression'
import './App.css'

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>逻辑回归演示</h1>
        <p>基于学习时间和出勤率预测考试通过概率</p>
      </header>
      <main className="app-main">
        <LogisticRegressionDemo />
      </main>
      <footer className="app-footer">
        <p>© 2024 统计学习演示项目</p>
      </footer>
    </div>
  )
}

export default App 