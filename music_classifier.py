
from flask import Flask, request, jsonify, render_template_string
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import lower, regexp_replace, col

app = Flask(__name__)

# Start Spark and load model once 
spark = SparkSession.builder \
    .appName("MusicGenrePredictor") \
    .master("local[2]") \
    .config("spark.driver.memory", "3g") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

model = PipelineModel.load("model/genre_model")

with open("data/labels.txt") as f:
    LABELS = [l.strip() for l in f.readlines()]

print(f"Model loaded. Genres: {LABELS}")

GENRE_COLORS = {
    "pop":     "#e94560",
    "rock":    "#e67e22",
    "country": "#f39c12",
    "blues":   "#3498db",
    "reggae":  "#2ecc71",
    "jazz":    "#9b59b6",
    "hip hop": "#1abc9c",
    "dance":   "#fd79a8",
}

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Music Genre Classifier</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #0f0f1a;
    color: #eee;
    min-height: 100vh;
    padding: 30px 20px;
  }
  .container { max-width: 860px; margin: 0 auto; }
  h1 {
    text-align: center;
    font-size: 2.4em;
    color: #fd79a8;
    margin-bottom: 6px;
    letter-spacing: 2px;
  }
  .subtitle {
    text-align: center;
    color: #888;
    margin-bottom: 30px;
    font-size: 0.95em;
  }
  .genres-list {
    text-align: center;
    margin-bottom: 24px;
    font-size: 0.85em;
    color: #aaa;
  }
  .genre-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    margin: 3px;
    font-size: 0.85em;
    font-weight: 600;
  }
  textarea {
    width: 100%;
    height: 200px;
    padding: 14px;
    font-size: 14px;
    line-height: 1.6;
    background: #1a1a2e;
    color: #eee;
    border: 2px solid #2a2a4e;
    border-radius: 10px;
    resize: vertical;
    outline: none;
    transition: border 0.2s;
  }
  textarea:focus { border-color: #fd79a8; }
  .btn {
    display: block;
    margin: 18px auto 0;
    padding: 14px 52px;
    font-size: 17px;
    font-weight: 700;
    background: linear-gradient(135deg, #e94560, #fd79a8);
    color: #fff;
    border: none;
    border-radius: 30px;
    cursor: pointer;
    transition: transform 0.15s, opacity 0.15s;
    letter-spacing: 1px;
  }
  .btn:hover { transform: scale(1.04); opacity: 0.92; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
  #loading {
    text-align: center;
    margin-top: 20px;
    color: #aaa;
    display: none;
    font-size: 15px;
  }
  #result {
    display: none;
    margin-top: 32px;
    background: #1a1a2e;
    border-radius: 14px;
    padding: 28px;
    border: 1px solid #2a2a3e;
  }
  #predicted {
    text-align: center;
    font-size: 2.2em;
    font-weight: 700;
    margin-bottom: 6px;
    letter-spacing: 2px;
  }
  #confidence {
    text-align: center;
    color: #aaa;
    margin-bottom: 24px;
    font-size: 0.95em;
  }
  .chart-wrap {
    position: relative;
    max-width: 640px;
    margin: 0 auto;
  }
  .top-three {
    display: flex;
    justify-content: center;
    gap: 16px;
    margin-bottom: 24px;
    flex-wrap: wrap;
  }
  .top-card {
    background: #0f0f1a;
    border-radius: 10px;
    padding: 12px 20px;
    text-align: center;
    min-width: 130px;
    border: 1px solid #2a2a3e;
  }
  .top-card .rank { font-size: 0.8em; color: #888; margin-bottom: 4px; }
  .top-card .name { font-size: 1.1em; font-weight: 700; margin-bottom: 4px; }
  .top-card .pct  { font-size: 1.4em; font-weight: 700; }
</style>
</head>
<body>
<div class="container">
  <h1>🎵 Music Genre Classifier</h1>
  <p class="subtitle">&nbsp; 8 Genre Classes</p>

  <div class="genres-list">
    {% for genre, color in genre_colors.items() %}
    <span class="genre-tag" style="background:{{ color }}22; color:{{ color }}; border:1px solid {{ color }}">
      {{ genre.upper() }}
    </span>
    {% endfor %}
  </div>

  <textarea id="lyricsBox"
    placeholder="Paste song lyrics here..."></textarea>

  <button class="btn" id="classifyBtn" onclick="classify()">
    🎯 &nbsp; Classify Genre
  </button>
  <p id="loading">⏳ Running Spark model... please wait</p>

  <div id="result">
    <div id="predicted"></div>
    <div id="confidence"></div>

    <div class="top-three" id="topThree"></div>

    <div class="chart-wrap">
      <canvas id="myChart"></canvas>
    </div>
  </div>
</div>

<script>
const COLORS = {{ colors | tojson }};
const LABELS = {{ labels | tojson }};
let chart = null;

async function classify() {
  const lyrics = document.getElementById('lyricsBox').value.trim();
  if (!lyrics) { alert('Please paste some lyrics first!'); return; }

  document.getElementById('classifyBtn').disabled = true;
  document.getElementById('loading').style.display = 'block';
  document.getElementById('result').style.display = 'none';

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lyrics })
    });
    const data = await resp.json();
    if (data.error) { alert('Error: ' + data.error); return; }

    // Set predicted genre with its color
    const predColor = COLORS[LABELS.indexOf(data.predicted_genre.toLowerCase())] || '#fd79a8';
    document.getElementById('predicted').innerHTML =
      `<span style="color:${predColor}">🏆 ${data.predicted_genre.toUpperCase()}</span>`;

    const topProb = data.probabilities[0].probability;
    document.getElementById('confidence').textContent =
      `Confidence: ${(topProb * 100).toFixed(1)}%`;

    // Top 3 cards
    const top3 = data.probabilities.slice(0, 3);
    const medals = ['🥇', '🥈', '🥉'];
    document.getElementById('topThree').innerHTML = top3.map((p, i) => {
      const c = COLORS[LABELS.indexOf(p.genre.toLowerCase())] || '#aaa';
      return `<div class="top-card" style="border-color:${c}44">
        <div class="rank">${medals[i]} #${i+1}</div>
        <div class="name" style="color:${c}">${p.genre.toUpperCase()}</div>
        <div class="pct" style="color:${c}">${(p.probability*100).toFixed(1)}%</div>
      </div>`;
    }).join('');

    // Bar chart
    const chartLabels = data.probabilities.map(p => p.genre.toUpperCase());
    const chartData   = data.probabilities.map(p => (p.probability * 100).toFixed(2));
    const chartColors = data.probabilities.map(p =>
      COLORS[LABELS.indexOf(p.genre.toLowerCase())] || '#aaa'
    );

    if (chart) chart.destroy();
    chart = new Chart(document.getElementById('myChart'), {
      type: 'bar',
      data: {
        labels: chartLabels,
        datasets: [{
          label: 'Probability (%)',
          data: chartData,
          backgroundColor: chartColors.map(c => c + 'cc'),
          borderColor: chartColors,
          borderWidth: 2,
          borderRadius: 8,
          borderSkipped: false
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          title: {
            display: true,
            text: 'Genre Probability Distribution',
            color: '#eee',
            font: { size: 15, weight: '600' }
          },
          tooltip: {
            callbacks: {
              label: ctx => ` ${ctx.parsed.y}%`
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            ticks: { color: '#ccc', callback: v => v + '%' },
            grid: { color: '#2a2a3e' }
          },
          x: {
            ticks: { color: '#ccc' },
            grid: { color: '#2a2a3e' }
          }
        }
      }
    });

    document.getElementById('result').style.display = 'block';

  } catch(e) {
    alert('Request failed: ' + e);
  } finally {
    document.getElementById('classifyBtn').disabled = false;
    document.getElementById('loading').style.display = 'none';
  }
}
</script>
</body>
</html>
"""

@app.route('/')
def index():
    genre_colors = {l: GENRE_COLORS.get(l.lower(), "#aaaaaa") for l in LABELS}
    colors = [GENRE_COLORS.get(l.lower(), "#aaaaaa") for l in LABELS]
    return render_template_string(HTML,
                                   genre_colors=genre_colors,
                                   colors=colors,
                                   labels=[l.lower() for l in LABELS])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        lyrics_text = request.get_json().get('lyrics', '')

        df = spark.createDataFrame([(lyrics_text,)], ["lyrics"])
        df = df.withColumn("lyrics", lower(col("lyrics")))
        df = df.withColumn("lyrics", regexp_replace(col("lyrics"), "[^a-z\\s]", " "))
        df = df.withColumn("lyrics", regexp_replace(col("lyrics"), "\\s+", " "))

        result = model.transform(df)
        row = result.select("predicted_genre", "probability").first()

        proba_arr = row["probability"].toArray()
        probs = [
            {"genre": LABELS[i], "probability": float(proba_arr[i])}
            for i in range(len(LABELS))
        ]
        probs.sort(key=lambda x: -x["probability"])

        return jsonify({
            "predicted_genre": row["predicted_genre"],
            "probabilities": probs
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
