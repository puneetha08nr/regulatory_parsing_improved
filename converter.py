# web_converter.py

from flask import Flask, request, jsonify, render_template_string
from auto_convert import ControlToJSONConverter

app = Flask(__name__)
converter = ControlToJSONConverter()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>UAE IA Control Converter</title>
    <style>
        body { font-family: Arial; margin: 40px; }
        textarea { width: 100%; height: 300px; font-family: monospace; }
        button { padding: 10px 20px; font-size: 16px; }
        pre { background: #f5f5f5; padding: 15px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>UAE IA Control to JSON Converter</h1>
    <p>Paste your control text below:</p>
    
    <textarea id="input"></textarea>
    <br><br>
    <button onclick="convert()">Convert to JSON</button>
    <button onclick="copyResult()">Copy JSON</button>
    
    <h2>Result:</h2>
    <pre id="output"></pre>
    
    <script>
        function convert() {
            const text = document.getElementById('input').value;
            
            fetch('/convert', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            })
            .then(r => r.json())
            .then(data => {
                document.getElementById('output').textContent = 
                    JSON.stringify(data, null, 2);
            });
        }
        
        function copyResult() {
            const text = document.getElementById('output').textContent;
            navigator.clipboard.writeText(text);
            alert('Copied to clipboard!');
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/convert', methods=['POST'])
def convert():
    data = request.json
    text = data.get('text', '')
    
    try:
        result = converter.parse_control_text(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)