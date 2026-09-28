import { useState } from "react"
import type { SubmitEvent } from "react"
import { Search, Sparkles } from "lucide-react"
import ReactMarkdown from "react-markdown"

function App() {
const [question, setQuestion] = useState("")
const [useWebSearch, setUseWebSearch] = useState(false)
const [topK, setTopK] = useState(5)
const [answer, setAnswer] = useState("")
const [sources, setSources] = useState<
  { source: string; title: string | null; url: string | null }[]
>([])
const [loading, setLoading] = useState(false)
const [error, setError] = useState("")
const handleSubmit = async (event: SubmitEvent) => {
  event.preventDefault()

  if (!question.trim()) {
    return
  }

  setLoading(true)
  setError("")
  setAnswer("")
  setSources([])

  try {
    const response = await fetch("http://127.0.0.1:8000/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: question.trim(),
        use_web_search: useWebSearch,
        top_k: topK,
      }),
    })

    const data = await response.json()

    if (!response.ok) {
      throw new Error(data.detail || "Research request failed.")
    }

    setAnswer(data.answer)
    setSources(data.sources)
  } catch (err) {
    setError(
      err instanceof Error ? err.message : "Research request failed.",
    )
  } finally {
    setLoading(false)
  }
}
  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      <aside className="fixed inset-y-0 left-0 w-64 border-r border-white/10 bg-[#0d0d0d]">
        <div className="flex h-full flex-col p-4">
          <div className="flex items-center gap-2 px-2 py-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-white text-black">
              <Sparkles size={16} />
            </div>
            <span className="font-semibold">Research AI</span>
          </div>

          <button className="mt-6 flex items-center gap-2 rounded-lg border border-white/10 px-3 py-2 text-sm text-white/70 transition hover:bg-white/5">
            <Search size={16} />
            New research
          </button>

          <div className="mt-8">
            <p className="px-2 text-xs font-medium uppercase tracking-wider text-white/30">
              Recent
            </p>

            <div className="mt-3 space-y-1">
              <button className="w-full rounded-lg px-3 py-2 text-left text-sm text-white/60 transition hover:bg-white/5 hover:text-white">
                AI research trends
              </button>
              <button className="w-full rounded-lg px-3 py-2 text-left text-sm text-white/60 transition hover:bg-white/5 hover:text-white">
                RAG architectures
              </button>
            </div>
          </div>
        </div>
      </aside>

      <main className="ml-64 min-h-screen">
        <div className="mx-auto flex min-h-screen max-w-5xl flex-col px-8 py-10">
          <header className="mb-16">
            <p className="text-sm text-white/40">AI Research Assistant · Built by Refih Can</p>
            <h1 className="mt-3 text-4xl font-semibold tracking-tight">
              What do you want to research?
            </h1>
            <p className="mt-3 max-w-2xl text-white/40">
              Ask a question and get an evidence-backed answer from your
              documents and the web.
            </p>
          </header>

          <form
              onSubmit={handleSubmit}
              className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 shadow-2xl shadow-black/20"
            >
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              className="min-h-36 w-full resize-none bg-transparent p-3 text-lg outline-none placeholder:text-white/20"
              placeholder="Ask anything..."
            />

            <div className="mt-4 flex items-center justify-between border-t border-white/10 pt-4">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setUseWebSearch((value) => !value)}
                  className={`rounded-lg px-3 py-2 text-sm transition ${
                    useWebSearch
                      ? "bg-white/10 text-white"
                      : "text-white/50 hover:bg-white/5 hover:text-white"
                  }`}
                >
                  Web search
                </button>

                <label className="flex items-center gap-2 text-sm text-white/50">
                  Top K
                  <input
                    type="number"
                    min="1"
                    value={topK}
                    onChange={(event) => setTopK(Number(event.target.value))}
                    className="w-16 rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-white outline-none"
                  />
                </label>
              </div>

              <button
                type="submit"
                disabled={loading || !question.trim()}
                className="rounded-lg bg-white px-4 py-2 text-sm font-medium text-black transition hover:bg-white/90 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {loading ? "Researching..." : "Research"}
              </button>
            </div>
          </form>

          <section className="mt-12">
            <div className="mb-4 flex items-center gap-2">
              <Sparkles size={16} className="text-white/50" />
              <h2 className="text-sm font-medium text-white/60">Answer</h2>
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-6">
              {error ? (
                <p className="text-sm text-red-400">{error}</p>
              ) : answer ? (
                <div className="prose prose-invert max-w-none text-white/70">
                  <ReactMarkdown>{answer}</ReactMarkdown>
                </div>
              ) : (
                <p className="leading-7 text-white/30">
                  Your research answer will appear here.
                </p>
              )}
            </div>
          </section>

          {sources.length > 0 ? (
            <div className="space-y-2">
              {sources.map((source, index) => (
                <div
                  key={`${source.source}-${index}`}
                  className="rounded-xl border border-white/10 bg-white/[0.02] p-4"
                >
                  <div className="mb-2 text-xs font-medium uppercase tracking-wider text-white/30">
                    Source {index + 1}
                  </div>
                  {source.url ? (
                    <a
                      href={source.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="font-medium text-white hover:underline"
                    >
                      {source.title || source.source}
                    </a>
                  ) : (
                    <p className="font-medium text-white">
                      {source.title || source.source}
                    </p>
                  )}

                  {source.title && source.title !== source.source && (
                    <p className="mt-1 text-sm text-white/30">
                      {source.source}
                    </p>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-6 text-sm text-white/30">
              Sources will appear here after research.
            </div>
          )}
        </div>
        <footer className="mt-10 border-t border-white/10 pt-6 text-center text-xs text-white/30">
          AI Research Assistant · Built by Refih
        </footer>
      </main>
    </div>
  )
}

export default App