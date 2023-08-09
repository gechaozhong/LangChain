from flask import Flask, request, jsonify
from langchain_llm35 import LangChainLLM35
app = Flask(__name__)

template = "{text}何许人也."
LangChainLLM = LangChainLLM35(template)


@app.route('/api/ai/convert', methods=['POST'])
def post_handle():
    try:
        data = request.json  # 获取POST请求中的JSON数据
        if data["content"]:
            content = data["content"]
            print(f"receive content: {content}")
            print(data["content"])
            ai_response = LangChainLLM.request(content)
            print(f"ai response: {ai_response}")
        # 在这里进行处理，可以根据需要对数据进行操作
        result = {"message": "Data handle successfully", "aiResponse": ai_response}
        response = jsonify(result)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response, 200
    except Exception as e:
        return jsonify({"error": "An error occurred", "details": str(e)}), 500


@app.route('/api/chat/completions', methods=['POST'])
def post_handle_simulation():
    try:
        simulation_result = "this is a simulation result"
        # 在这里进行处理，可以根据需要对数据进行操作
        result = {"message": "Data handle successfully", "result": "this is a test case",
                  "choices": [
                      {"message": {"role": "user", "content": simulation_result, "name": "name1"}}
                  ]}
        response = jsonify(result)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response, 200
    except Exception as e:
        return jsonify({"error": "An error occurred", "details": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
